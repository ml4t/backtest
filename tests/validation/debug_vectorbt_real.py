"""
Debug script to verify we're using real VectorBT, not fake implementation
"""

import sys

import numpy as np
import pandas as pd


def verify_vectorbt_real():
    """Verify we're using the actual VectorBT library."""
    print("Verifying VectorBT Implementation")
    print("=" * 50)

    try:
        import vectorbt as vbt

        print("✓ VectorBT imported successfully")
        print(f"  Version: {getattr(vbt, '__version__', 'unknown')}")
        print(f"  Module path: {vbt.__file__}")
        print(
            f"  Available attributes: {sorted([attr for attr in dir(vbt) if not attr.startswith('_')])}",
        )

        # Create simple test data
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        np.random.seed(42)
        close = pd.Series(100 * (1 + np.random.normal(0, 0.01, 100)).cumprod(), index=dates)

        print("\nTesting VectorBT functionality:")
        print(f"  Close data shape: {close.shape}")

        # Test moving average
        print("\n1. Testing MA indicator...")
        try:
            ma_result = vbt.MA.run(close, window=20)
            print("  ✓ MA.run() successful")
            print(f"  MA object type: {type(ma_result)}")
            print(
                f"  MA object attributes: {sorted([attr for attr in dir(ma_result) if not attr.startswith('_')])}",
            )

            ma_values = ma_result.ma
            print(f"  MA values type: {type(ma_values)}")
            print(f"  MA values shape: {ma_values.shape}")
            print(f"  First 5 MA values: {ma_values.head()}")

        except Exception as e:
            print(f"  ✗ MA test failed: {e}")
            return False

        # Test signal generation
        print("\n2. Testing signal generation...")
        try:
            ma_short = vbt.MA.run(close, window=10).ma
            ma_long = vbt.MA.run(close, window=20).ma

            entries = (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))
            exits = (ma_short <= ma_long) & (ma_short.shift(1) > ma_long.shift(1))

            print("  ✓ Signal generation successful")
            print(f"  Entries count: {entries.sum()}")
            print(f"  Exits count: {exits.sum()}")

        except Exception as e:
            print(f"  ✗ Signal generation failed: {e}")
            return False

        # Test portfolio creation
        print("\n3. Testing Portfolio.from_signals...")
        try:
            pf = vbt.Portfolio.from_signals(
                close,
                entries=entries,
                exits=exits,
                init_cash=10000,
                fees=0.0,
            )
            print("  ✓ Portfolio created successfully")
            print(f"  Portfolio type: {type(pf)}")
            print(
                f"  Portfolio attributes: {sorted([attr for attr in dir(pf) if not attr.startswith('_') and not callable(getattr(pf, attr))])}",
            )

            # Test portfolio metrics
            final_value = pf.final_value()
            total_return = pf.total_return()

            print(f"  Final value: {final_value} (type: {type(final_value)})")
            print(f"  Total return: {total_return} (type: {type(total_return)})")

            # Test orders/trades
            if hasattr(pf, "orders"):
                print(
                    f"  Orders available: {len(pf.orders.records) if hasattr(pf.orders, 'records') else 'unknown'}",
                )

            if hasattr(pf, "trades"):
                print(f"  Trades available: {hasattr(pf.trades, 'records')}")
                if hasattr(pf.trades, "records_readable"):
                    trades_df = pf.trades.records_readable
                    print(f"  Trades dataframe shape: {trades_df.shape}")
                    print(f"  Trades columns: {list(trades_df.columns)}")

        except Exception as e:
            print(f"  ✗ Portfolio test failed: {e}")
            import traceback

            traceback.print_exc()
            return False

        print("\n✅ VectorBT verification successful - using real implementation!")
        return True

    except ImportError as e:
        print(f"✗ VectorBT not available: {e}")
        return False
    except Exception as e:
        print(f"✗ VectorBT verification failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_vectorbt_real()
    if not success:
        print("\n❌ VectorBT verification failed!")
        sys.exit(1)
    else:
        print("\n✅ VectorBT verification passed!")
