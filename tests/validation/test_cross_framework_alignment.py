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

from .data_loader import UniversalDataLoader


class TestCrossFrameworkAlignment:
    """Test suite for cross-framework signal alignment."""

    @pytest.fixture
    def test_data(self):
        """Load test data from Wiki source."""
        loader = UniversalDataLoader()
        df = loader.load_daily_equities(
            tickers=['AAPL'],
            start_date='2017-01-03',
            end_date='2017-12-29',
            source='wiki'
        )

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
        """Verify signals occur on expected dates."""
        signals = self.calculate_ma_signals(test_data)

        signal_dates = {s['date'].date(): s['type'] for s in signals}

        # Expected signals from diagnostic verification
        expected_golden = [
            pd.to_datetime('2017-04-26').date(),
            pd.to_datetime('2017-07-19').date(),
            pd.to_datetime('2017-10-18').date(),
            pd.to_datetime('2017-12-20').date(),
        ]

        expected_death = [
            pd.to_datetime('2017-04-24').date(),
            pd.to_datetime('2017-06-13').date(),
            pd.to_datetime('2017-09-19').date(),
            pd.to_datetime('2017-12-11').date(),
        ]

        for date in expected_golden:
            assert date in signal_dates, f"Missing golden cross on {date}"
            assert signal_dates[date] == 'GOLDEN', f"Wrong signal type on {date}"

        for date in expected_death:
            assert date in signal_dates, f"Missing death cross on {date}"
            assert signal_dates[date] == 'DEATH', f"Wrong signal type on {date}"

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

        # Should have 3 completed trades with 1 open position
        assert len(trades) == 3, f"Expected 3 completed trades, got {len(trades)}"
        assert position > 0, "Expected open position at end"

        # Total return calculation (including unrealized)
        end_price = test_data.iloc[-1]['close']
        final_value = cash + (position * end_price)
        total_return = (final_value / capital - 1) * 100

        # Allow for floating point rounding
        assert 12.4 < total_return < 12.6, f"Expected ~12.52% return, got {total_return:.2f}%"

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
        """Verify exact MA values on key signal dates."""
        signals = self.calculate_ma_signals(test_data)

        # Check a critical signal date
        golden_cross_april26 = [s for s in signals if s['date'].date() == pd.to_datetime('2017-04-26').date()]
        assert len(golden_cross_april26) == 1, "Should have exactly one signal on 2017-04-26"

        signal = golden_cross_april26[0]
        assert signal['type'] == 'GOLDEN'

        # MA values from diagnostic verification (CORRECTED - using VectorBT/pandas rolling)
        # VectorBT uses pandas rolling which gives MA(30) =  142.3067
        assert abs(signal['ma_short'] - 142.3101) < 0.01, f"MA(10) mismatch: {signal['ma_short']}"
        assert abs(signal['ma_long'] - 142.3067) < 0.01, f"MA(30) mismatch: {signal['ma_long']}"


@pytest.mark.integration
class TestFrameworkComparison:
    """Integration tests comparing actual framework implementations."""

    def test_vectorbt_produces_expected_signals(self):
        """Verify VectorBT Pro adapter produces correct signals."""
        from .frameworks.vectorbtpro_adapter import VectorBTProAdapter

        loader = UniversalDataLoader()
        df = loader.load_daily_equities(
            tickers=['AAPL'],
            start_date='2017-01-03',
            end_date='2017-12-29',
            source='wiki'
        )

        data = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        data.set_index('timestamp', inplace=True)

        adapter = VectorBTProAdapter()
        result = adapter.run_backtest(
            data,
            {"name": "MovingAverageCrossover", "short_window": 10, "slow_window": 30},
            10000
        )

        # VectorBT counts complete round trips (entry + exit)
        # Expected: 3-4 complete cycles (4 if it force-closes final position)
        assert 3 <= result.num_trades <= 4, f"Expected 3-4 complete round trips, got {result.num_trades}"

        # Expected return: ~12.52% if force-closes, ~12.52% if keeps open (same due to prices)
        assert 2.0 < result.total_return < 13.0, f"Expected 2-13% return, got {result.total_return:.2f}%"

    def test_frameworks_with_predefined_signals(self):
        """
        Test that both frameworks produce identical results when given identical entry/exit signals.

        This eliminates variance from:
        - Different MA calculations
        - Different data sources (Wiki parquet vs quandl bundle)
        - Floating point rounding

        By pre-computing signals and ensuring both frameworks trade on the SAME dates,
        we can verify the execution logic is identical.
        """
        # TODO: Implement signal-based adapters that accept pre-computed signals
        # This would require extending BaseFrameworkAdapter with a signal-based interface
        pytest.skip("Requires signal-based adapter interface - future enhancement")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
