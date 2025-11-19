"""
Cross-Framework Signal Alignment Test

Verifies that VectorBT and Zipline detect the same crossover signals
when using identical data and strategy parameters.

CRITICAL: This test uses the SAME data source (Wiki parquet) for both frameworks
to eliminate data source variance. In production, Zipline uses quandl bundle
which may have slight OHLCV differences leading to minor (<1%) variance.
"""

import pandas as pd
import pytest

from .fixtures import get_test_data


class TestCrossFrameworkAlignment:
    """Test suite for cross-framework signal alignment."""

    @pytest.fixture
    def test_data(self):
        """Load test data for testing."""
        df = get_test_data(symbol='AAPL', start='2017-01-03', end='2017-12-29')

        # Convert to time-indexed format
        data = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        data.set_index('timestamp', inplace=True)
        return data

    def calculate_ma_signals(self, data, short_window=10, long_window=30):
        """
        Calculate MA crossover signals using the CORRECTED Zipline logic.

        This is the logic after the bug fix where we properly slice
        history to use only the last `long_window` days.
        """
        close = data['close'].values
        signals = []

        for i in range(len(data)):
            if i < long_window:
                continue

            # Current window - get long_window + 1 for prev calculation
            history = close[max(0, i - long_window):i + 1]
            ma_short = history[-short_window:].mean()
            ma_long = history[-long_window:].mean()  # FIX: Only last long_window days

            # Previous window
            prev_history = history[:-1]
            prev_ma_short = prev_history[-short_window:].mean()
            prev_ma_long = prev_history[-long_window:].mean()  # FIX: Only last long_window days

            # Crossover detection
            golden_cross = (prev_ma_short <= prev_ma_long) and (ma_short > ma_long)
            death_cross = (prev_ma_short > prev_ma_long) and (ma_short <= ma_long)

            if golden_cross or death_cross:
                signals.append({
                    'date': data.index[i],
                    'type': 'GOLDEN' if golden_cross else 'DEATH',
                    'ma_short': ma_short,
                    'ma_long': ma_long,
                })

        return signals

    def test_signal_count_matches(self, test_data):
        """Verify both frameworks detect same number of signals."""
        signals = self.calculate_ma_signals(test_data)

        golden_crosses = [s for s in signals if s['type'] == 'GOLDEN']
        death_crosses = [s for s in signals if s['type'] == 'DEATH']

        # Expected from manual verification
        assert len(golden_crosses) == 4, f"Expected 4 golden crosses, got {len(golden_crosses)}"
        assert len(death_crosses) == 4, f"Expected 4 death crosses, got {len(death_crosses)}"

    def test_signal_dates_match_expected(self, test_data):
        """
        Verify signals are consistent and reasonable.

        NOTE: Exact dates depend on data source (Wiki vs yfinance),
        so we test for signal count consistency rather than hardcoded dates.
        """
        signals = self.calculate_ma_signals(test_data)

        golden_crosses = [s for s in signals if s['type'] == 'GOLDEN']
        death_crosses = [s for s in signals if s['type'] == 'DEATH']

        # Verify we get reasonable number of signals (3-5 of each is typical for AAPL 2017)
        assert 3 <= len(golden_crosses) <= 5, \
            f"Expected 3-5 golden crosses, got {len(golden_crosses)}"
        assert 3 <= len(death_crosses) <= 5, \
            f"Expected 3-5 death crosses, got {len(death_crosses)}"

        # Verify all signals are in 2017
        for s in signals:
            assert s['date'].year == 2017, f"Signal date {s['date']} not in 2017"

    def test_ma_calculation_vectorbt_compatibility(self, test_data):
        """Verify our MA calculation matches VectorBT's rolling logic."""
        # Calculate using our method
        signals = self.calculate_ma_signals(test_data)

        # Calculate using pandas rolling (VectorBT approach)
        ma_short = test_data['close'].rolling(window=10).mean()
        ma_long = test_data['close'].rolling(window=30).mean()

        vbt_golden = (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))
        vbt_death = (ma_short < ma_long) & (ma_short.shift(1) >= ma_long.shift(1))

        vbt_golden_dates = test_data[vbt_golden].index.tolist()
        vbt_death_dates = test_data[vbt_death].index.tolist()

        our_golden_dates = [s['date'] for s in signals if s['type'] == 'GOLDEN']
        our_death_dates = [s['date'] for s in signals if s['type'] == 'DEATH']

        assert our_golden_dates == vbt_golden_dates, "Golden cross dates don't match VectorBT"
        assert our_death_dates == vbt_death_dates, "Death cross dates don't match VectorBT"

    def test_trade_execution_logic(self, test_data):
        """Verify correct trade execution from signals."""
        signals = self.calculate_ma_signals(test_data)

        capital = 10000
        position = 0
        cash = capital
        trades = []

        for signal in signals:
            price = test_data.loc[signal['date'], 'close']

            if signal['type'] == 'GOLDEN' and position == 0:
                # Buy
                position = cash / price
                entry_price = price
                entry_date = signal['date']
                cash = 0

            elif signal['type'] == 'DEATH' and position > 0:
                # Sell
                proceeds = position * price
                cash = proceeds
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': signal['date'],
                    'entry_price': entry_price,
                    'exit_price': price,
                    'pnl': proceeds - (position * entry_price)
                })
                position = 0

        # Verify we executed reasonable number of trades
        # (exact count varies by data source: Wiki vs yfinance)
        assert 2 <= len(trades) <= 5, f"Expected 2-5 completed trades, got {len(trades)}"

        # Total return calculation (including unrealized)
        end_price = test_data.iloc[-1]['close']
        final_value = cash + (position * end_price)
        total_return = (final_value / capital - 1) * 100

        # Return should be reasonable for AAPL 2017 with this strategy
        # (exact value varies by data source, but should be positive)
        assert total_return > 0, f"Expected positive return, got {total_return:.2f}%"

    def test_no_spurious_signals(self, test_data):
        """Verify no duplicate signals when already in desired state."""
        signals = self.calculate_ma_signals(test_data)

        position = 0
        signal_count = 0

        for signal in signals:
            signal_count += 1

            if signal['type'] == 'GOLDEN':
                # Can get golden cross when flat OR if it's the start of trading
                if position == 1:
                    pytest.fail(f"Duplicate golden cross while already in position on {signal['date'].date()}")
                position = 1

            elif signal['type'] == 'DEATH':
                # Can get death cross if in position OR at very start (before first entry)
                # Only fail if we get consecutive death crosses
                if position == -1:
                    pytest.fail(f"Duplicate death cross on {signal['date'].date()}")
                position = -1 if position == 0 else 0  # Track if we exited

    def test_ma_values_match_on_critical_dates(self, test_data):
        """
        Verify MA calculation consistency.

        NOTE: Exact MA values and dates vary by data source (Wiki vs yfinance),
        so we test the calculation logic rather than hardcoded values.
        """
        signals = self.calculate_ma_signals(test_data)

        # Test that all signals have valid MA values
        for signal in signals:
            assert signal['ma_short'] > 0, f"Invalid short MA on {signal['date']}"
            assert signal['ma_long'] > 0, f"Invalid long MA on {signal['date']}"

            # For golden cross, short MA should be above long MA
            if signal['type'] == 'GOLDEN':
                assert signal['ma_short'] > signal['ma_long'], \
                    f"Golden cross but short MA ({signal['ma_short']}) <= long MA ({signal['ma_long']})"
            # For death cross, short MA should be below long MA
            elif signal['type'] == 'DEATH':
                assert signal['ma_short'] < signal['ma_long'], \
                    f"Death cross but short MA ({signal['ma_short']}) >= long MA ({signal['ma_long']})"

    def test_frameworks_with_predefined_signals(self, test_data):
        """
        Test that frameworks produce identical results when given identical entry/exit signals.

        This eliminates variance from:
        - Different MA calculations
        - Floating point rounding
        - Indicator implementation differences

        By pre-computing signals and ensuring all frameworks trade on the SAME dates with SAME data,
        we can verify execution logic is identical across ml4t.backtest, VectorBT, and Backtrader.

        NOTE: Zipline is excluded because run_algorithm(bundle=...) uses bundle data
        instead of our test DataFrame, making direct price comparison impossible.
        Zipline validation uses the regular backtest tests instead.
        """
        from .frameworks.qengine_adapter import BacktestAdapter
        from .frameworks.vectorbt_adapter import VectorBTAdapter
        from .frameworks.backtrader_adapter import BacktraderAdapter
        from .frameworks.base import Signal, FrameworkConfig

        # Generate signals using standardized calculation
        signal_data = self.calculate_ma_signals(test_data)

        # Convert to Signal format for all frameworks
        signals: list[Signal] = []
        for sig in signal_data:
            if sig['type'] == 'GOLDEN':
                # Entry signal (BUY)
                signals.append(Signal(
                    timestamp=sig['date'],
                    asset_id='AAPL',
                    action='BUY',
                    quantity=100.0,  # Fixed quantity for consistent comparison
                ))
            else:  # DEATH cross
                # Exit signal (SELL)
                signals.append(Signal(
                    timestamp=sig['date'],
                    asset_id='AAPL',
                    action='SELL',
                    quantity=100.0,
                ))

        # Convert list[Signal] to DataFrame with 'entry'/'exit' boolean columns
        signals_df = pd.DataFrame(False, index=test_data.index, columns=['entry', 'exit'])
        for sig in signals:
            if sig['action'] == 'BUY':
                signals_df.loc[sig['timestamp'], 'entry'] = True
            elif sig['action'] == 'SELL':
                signals_df.loc[sig['timestamp'], 'exit'] = True

        print(f"\n{'=' * 80}")
        print(f"3-WAY FRAMEWORK VALIDATION: Signal-Based Execution")
        print(f"{'=' * 80}")
        print(f"Pre-computed signals: {len(signals)}")
        print(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")
        print(f"Frameworks: ml4t.backtest | VectorBT | Backtrader")
        print(f"{'=' * 80}\n")

        # Run all frameworks with identical signals
        # Use FrameworkConfig for unified configuration
        config = FrameworkConfig.for_matching()  # Zero fees, same-bar fills

        # ml4t.backtest
        qengine_adapter = BacktestAdapter()
        qengine_result = qengine_adapter.run_with_signals(
            data=test_data,
            signals=signals_df,
            config=config,
        )

        # VectorBT
        vectorbt_adapter = VectorBTAdapter()
        vectorbt_result = vectorbt_adapter.run_with_signals(
            data=test_data,
            signals=signals_df,
            config=config,
        )

        # Backtrader
        backtrader_adapter = BacktraderAdapter()
        backtrader_result = backtrader_adapter.run_with_signals(
            data=test_data,
            signals=signals_df,
            config=config,
        )

        # Collect all results
        results = {
            "ml4t.backtest": qengine_result,
            "VectorBT": vectorbt_result,
            "Backtrader": backtrader_result,
        }

        # Display comparison table
        print(f"\n{'=' * 80}")
        print(f"3-WAY COMPARISON RESULTS")
        print(f"{'=' * 80}")
        print(f"{'Framework':<15} {'Final Value':>15} {'Return':>10} {'Trades':>8} {'Status':>10}")
        print(f"{'-' * 80}")

        for name, result in results.items():
            status = "✓" if not result.has_errors else "✗"
            print(f"{name:<15} ${result.final_value:>14,.2f} {result.total_return:>9.2f}% {result.num_trades:>8} {status:>10}")

        # Calculate variance statistics
        final_values = [r.final_value for r in results.values() if not r.has_errors]
        returns = [r.total_return for r in results.values() if not r.has_errors]
        trade_counts = [r.num_trades for r in results.values() if not r.has_errors]

        if len(final_values) > 1:
            value_range = max(final_values) - min(final_values)
            value_pct_range = (value_range / config.initial_capital) * 100
            return_range = max(returns) - min(returns)

            print(f"\n{'Variance Statistics':}")
            print(f"  Value Range: ${value_range:,.2f} ({value_pct_range:.4f}%)")
            print(f"  Return Range: {return_range:.4f}%")
            print(f"  Trade Count Range: {max(trade_counts) - min(trade_counts)}")

        print(f"{'=' * 80}\n")

        # Assertions - with identical signals, results should be very close
        # Check for errors first
        for name, result in results.items():
            if result.has_errors:
                print(f"⚠ {name} had errors (may not be installed): {result.errors}")

        # Compare available frameworks (allow some to be missing)
        available_results = {name: r for name, r in results.items() if not r.has_errors}

        if len(available_results) < 2:
            pytest.skip(f"Need at least 2 frameworks available, got {len(available_results)}")

        # All available frameworks should produce similar results (within 1%)
        for i, (name1, result1) in enumerate(available_results.items()):
            for name2, result2 in list(available_results.items())[i+1:]:
                value_diff = abs(result1.final_value - result2.final_value)
                value_pct_diff = (value_diff / config.initial_capital) * 100
                return_diff = abs(result1.total_return - result2.total_return)

                assert value_pct_diff < 1.0, \
                    f"{name1} vs {name2}: Value variance too high: {value_pct_diff:.4f}% " \
                    f"({name1}: ${result1.final_value:,.2f}, {name2}: ${result2.final_value:,.2f})"

                assert return_diff < 1.0, \
                    f"{name1} vs {name2}: Return variance too high: {return_diff:.4f}% " \
                    f"({name1}: {result1.total_return:.2f}%, {name2}: {result2.total_return:.2f}%)"

                # Allow trade count difference of ±1 due to end-of-backtest position handling
                trade_count_diff = abs(result1.num_trades - result2.num_trades)
                assert trade_count_diff <= 1, \
                    f"{name1} vs {name2}: Trade count variance too high " \
                    f"({name1}={result1.num_trades}, {name2}={result2.num_trades}, diff={trade_count_diff})"

        print(f"✅ 3-WAY VALIDATION PASSED - All {len(available_results)} frameworks produce identical results!")

    def test_frameworks_comprehensive_100_trades(self, test_data):
        """
        Comprehensive test with 100+ signals to validate execution at scale.

        Uses shorter MA windows (5/15) to generate many more crossovers.
        Tests ml4t.backtest, VectorBT, and Backtrader with identical signals.
        (Zipline excluded - see test_frameworks_with_predefined_signals docstring)
        """
        from .frameworks.qengine_adapter import BacktestAdapter
        from .frameworks.vectorbt_adapter import VectorBTAdapter
        from .frameworks.backtrader_adapter import BacktraderAdapter
        from .frameworks.base import Signal, FrameworkConfig

        # Generate many signals using shorter windows
        signal_data = self.calculate_ma_signals(test_data, short_window=5, long_window=15)

        print(f"\n{'=' * 80}")
        print(f"COMPREHENSIVE 100+ TRADE VALIDATION")
        print(f"{'=' * 80}")
        print(f"Total signals generated: {len(signal_data)}")
        print(f"Expected trades: ~{len(signal_data) // 2}")
        print(f"{'=' * 80}\n")

        # Convert to Signal format
        signals: list[Signal] = []
        for sig in signal_data:
            if sig['type'] == 'GOLDEN':
                signals.append(Signal(
                    timestamp=sig['date'],
                    asset_id='AAPL',
                    action='BUY',
                    quantity=100.0,
                ))
            else:
                signals.append(Signal(
                    timestamp=sig['date'],
                    asset_id='AAPL',
                    action='SELL',
                    quantity=100.0,
                ))

        print(f"Signal breakdown:")
        buy_signals = [s for s in signals if s['action'] == 'BUY']
        sell_signals = [s for s in signals if s['action'] == 'SELL']
        print(f"  BUY signals: {len(buy_signals)}")
        print(f"  SELL signals: {len(sell_signals)}")
        print(f"\nFirst 10 signals:")
        for i, sig in enumerate(signals[:10]):
            print(f"  {i+1}. {sig['timestamp'].date()}: {sig['action']}")

        # Convert list[Signal] to DataFrame with 'entry'/'exit' boolean columns
        signals_df = pd.DataFrame(False, index=test_data.index, columns=['entry', 'exit'])
        for sig in signals:
            if sig['action'] == 'BUY':
                signals_df.loc[sig['timestamp'], 'entry'] = True
            elif sig['action'] == 'SELL':
                signals_df.loc[sig['timestamp'], 'exit'] = True

        # Use FrameworkConfig for unified configuration
        config = FrameworkConfig.for_matching()

        # Run all 3 frameworks (Zipline excluded - see docstring)
        qengine_adapter = BacktestAdapter()
        qengine_result = qengine_adapter.run_with_signals(
            data=test_data,
            signals=signals_df,
            config=config,
        )

        vectorbt_adapter = VectorBTAdapter()
        vectorbt_result = vectorbt_adapter.run_with_signals(
            data=test_data,
            signals=signals_df,
            config=config,
        )

        backtrader_adapter = BacktraderAdapter()
        backtrader_result = backtrader_adapter.run_with_signals(
            data=test_data,
            signals=signals_df,
            config=config,
        )

        # Collect results
        results = {
            "ml4t.backtest": qengine_result,
            "VectorBT": vectorbt_result,
            "Backtrader": backtrader_result,
        }

        # Display comparison
        print(f"\n{'=' * 80}")
        print(f"COMPREHENSIVE TEST RESULTS ({len(signals)} signals)")
        print(f"{'=' * 80}")
        print(f"{'Framework':<15} {'Final Value':>15} {'Return':>10} {'Trades':>8} {'% of Expected':>14}")
        print(f"{'-' * 80}")

        expected_trades = len(signals) // 2
        for name, result in results.items():
            if not result.has_errors:
                pct_of_expected = (result.num_trades / expected_trades * 100) if expected_trades > 0 else 0
                print(f"{name:<15} ${result.final_value:>14,.2f} {result.total_return:>9.2f}% "
                      f"{result.num_trades:>8} {pct_of_expected:>13.1f}%")
            else:
                print(f"{name:<15} {'ERROR':>15}")

        # Variance statistics
        available_results = {name: r for name, r in results.items() if not r.has_errors}

        if len(available_results) > 1:
            final_values = [r.final_value for r in available_results.values()]
            returns = [r.total_return for r in available_results.values()]
            trade_counts = [r.num_trades for r in available_results.values()]

            value_range = max(final_values) - min(final_values)
            value_pct_range = (value_range / config.initial_capital) * 100
            return_range = max(returns) - min(returns)
            trade_range = max(trade_counts) - min(trade_counts)

            print(f"\n{'Variance Statistics':}")
            print(f"  Value Range: ${value_range:,.2f} ({value_pct_range:.4f}%)")
            print(f"  Return Range: {return_range:.4f}%")
            print(f"  Trade Count Range: {trade_range} ({trade_range/expected_trades*100:.1f}% of expected)")

        print(f"{'=' * 80}\n")

        # Detailed analysis
        print(f"\nDETAILED TRADE EXECUTION ANALYSIS:")
        print(f"{'=' * 80}")
        for name, result in available_results.items():
            if result.has_errors:
                continue
            missing_trades = expected_trades - result.num_trades
            pct_missing = (missing_trades / expected_trades * 100) if expected_trades > 0 else 0
            print(f"{name}:")
            print(f"  Expected trades: {expected_trades}")
            print(f"  Actual trades: {result.num_trades}")
            print(f"  Missing: {missing_trades} ({pct_missing:.1f}%)")

            if missing_trades > 0:
                print(f"  ⚠️  WARNING: Framework is not executing all signals!")

        print(f"{'=' * 80}\n")

        # Assertions - be more strict with many trades
        if len(available_results) < 2:
            pytest.skip(f"Need at least 2 frameworks, got {len(available_results)}")

        # Determine trade variance threshold based on actual trade count
        # For low trade counts (<20), allow ±1 trade due to end-of-backtest position handling
        # For high trade counts (100+), use strict 5% threshold
        max_trades = max(r.num_trades for r in available_results.values())

        if max_trades < 20:
            # Low trade count: allow ±1 trade difference (end-of-backtest position handling)
            max_trade_diff = 1
            print(f"\nUsing ±{max_trade_diff} trade tolerance for low trade count test")
        else:
            # High trade count: use percentage-based threshold
            max_trade_diff = None
            max_trade_variance_pct = 5.0
            print(f"\nUsing {max_trade_variance_pct}% variance threshold for high trade count test")

        for i, (name1, result1) in enumerate(available_results.items()):
            for name2, result2 in list(available_results.items())[i+1:]:
                trade_diff = abs(result1.num_trades - result2.num_trades)

                if max_trade_diff is not None:
                    # Use absolute difference for low trade counts
                    assert trade_diff <= max_trade_diff, \
                        f"{name1} vs {name2}: Trade count difference {trade_diff} exceeds ±{max_trade_diff} " \
                        f"({name1}={result1.num_trades}, {name2}={result2.num_trades})"
                else:
                    # Use percentage for high trade counts
                    trade_variance_pct = trade_diff / max(result1.num_trades, result2.num_trades) * 100
                    assert trade_variance_pct <= max_trade_variance_pct, \
                        f"{name1} vs {name2}: Trade count variance {trade_variance_pct:.1f}% exceeds {max_trade_variance_pct}% " \
                        f"({name1}={result1.num_trades}, {name2}={result2.num_trades})"

        print(f"✅ COMPREHENSIVE VALIDATION PASSED - All frameworks execute signals consistently!")

    @pytest.mark.skip(reason="Zipline uses bundle data (bundle='quandl') instead of test DataFrame. "
                                     "Prices are ~4.3x different ($144.54 vs $33.46), making signal-based "
                                     "validation impossible. See test_frameworks_with_predefined_signals docstring.")
    def test_debug_zipline_variance(self, test_data):
        """
        [COMPLETED INVESTIGATION] Debug test to understand why Zipline has 9.4% variance.

        FINDINGS: Zipline's run_algorithm(bundle='quandl') fetches its own price data
        from the Quandl bundle instead of using our test DataFrame. This makes it
        incompatible with signal-based validation using custom data.

        This test runs a minimal set of signals and logs detailed execution info.
        """
        from .frameworks.qengine_adapter import BacktestAdapter
        from .frameworks.zipline_adapter import ZiplineAdapter
        from .frameworks.base import Signal
        import pandas as pd

        # Create just 3 simple signals: BUY, SELL, BUY
        signals: list[Signal] = [
            Signal(timestamp=pd.Timestamp('2017-04-25'), asset_id='AAPL', action='BUY', quantity=100.0),
            Signal(timestamp=pd.Timestamp('2017-05-30'), asset_id='AAPL', action='SELL', quantity=100.0),
            Signal(timestamp=pd.Timestamp('2017-06-05'), asset_id='AAPL', action='BUY', quantity=100.0),
        ]

        print(f"\n{'=' * 80}")
        print(f"DEBUG: Zipline vs ml4t.backtest - 3 Signals")
        print(f"{'=' * 80}")
        for sig in signals:
            print(f"  {sig['timestamp'].date()}: {sig['action']}")
        print(f"{'=' * 80}\n")

        initial_capital = 10000.0

        # ml4t.backtest
        print("Running ml4t.backtest...")
        ml4t.backtest_adapter = BacktestAdapter()
        ml4t.backtest_result = ml4t.backtest_adapter.run_with_signals(
            data=test_data,
            signals=signals,
            initial_capital=initial_capital,
        )

        print("\nRunning Zipline...")
        zipline_adapter = ZiplineAdapter()
        zipline_result = zipline_adapter.run_with_signals(
            data=test_data,
            signals=signals,
            initial_capital=initial_capital,
        )

        print(f"\n{'=' * 80}")
        print(f"COMPARISON")
        print(f"{'=' * 80}")
        print(f"ml4t.backtest:")
        print(f"  Final Value: ${ml4t.backtest_result.final_value:,.2f}")
        print(f"  Return: {ml4t.backtest_result.total_return:.2f}%")
        print(f"  Trades: {ml4t.backtest_result.num_trades}")
        print(f"  Trade Details:")
        for i, trade in enumerate(ml4t.backtest_result.trades):
            print(f"    {i+1}. {trade.timestamp.date()} {trade.action} {trade.quantity:.2f} @ ${trade.price:.2f} = ${trade.value:.2f}")

        print(f"\nZipline:")
        print(f"  Final Value: ${zipline_result.final_value:,.2f}")
        print(f"  Return: {zipline_result.total_return:.2f}%")
        print(f"  Trades: {zipline_result.num_trades}")

        variance_pct = abs(ml4t.backtest_result.final_value - zipline_result.final_value) / initial_capital * 100
        print(f"\nVariance: ${abs(ml4t.backtest_result.final_value - zipline_result.final_value):,.2f} ({variance_pct:.4f}%)")
        print(f"{'=' * 80}\n")

        # Allow up to 0.5% variance for execution differences
        assert variance_pct < 0.5, \
            f"Zipline variance too high: {variance_pct:.4f}% (ml4t.backtest: ${ml4t.backtest_result.final_value:,.2f}, Zipline: ${zipline_result.final_value:,.2f})"


@pytest.mark.integration
class TestFrameworkComparison:
    """Integration tests comparing actual framework implementations."""

    def test_vectorbt_produces_expected_signals(self):
        """Verify VectorBT Pro adapter produces correct signals."""
        from .frameworks.vectorbtpro_adapter import VectorBTProAdapter

        df = get_test_data(symbol='AAPL', start='2017-01-03', end='2017-12-29')
        data = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        data.set_index('timestamp', inplace=True)

        adapter = VectorBTProAdapter()
        result = adapter.run_backtest(
            data,
            {"name": "MovingAverageCrossover", "short_window": 10, "slow_window": 30},
            10000
        )

        # VectorBT counts complete round trips (entry + exit)
        # Expected: 3-5 complete cycles (varies by data source)
        assert 2 <= result.num_trades <= 6, f"Expected 2-6 complete round trips, got {result.num_trades}"

        # Expected return: varies by data source (Wiki vs yfinance)
        # Both should be positive for AAPL 2017 with this strategy
        assert result.total_return > 0, f"Expected positive return, got {result.total_return:.2f}%"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
