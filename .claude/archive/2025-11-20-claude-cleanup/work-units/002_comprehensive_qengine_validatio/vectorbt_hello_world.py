#!/usr/bin/env python
"""VectorBT Pro Hello World Test - Minimal functionality verification"""

# Disable auto-import to avoid loading missing data module
import os
os.environ['VBT_AUTO_IMPORT'] = '0'

# Try minimal import
try:
    from vectorbtpro._version import __version__
    print(f"✅ VectorBT Pro version: {__version__}")
    print("✅ Minimal import successful!")

    # Try importing core portfolio functionality (doesn't require data module)
    from vectorbtpro.portfolio import nb as pf_nb
    print("✅ Portfolio module accessible")

    # Try importing signals
    from vectorbtpro.signals import nb as sig_nb
    print("✅ Signals module accessible")

    # Try basic numba functions
    import numpy as np
    from vectorbtpro.generic import nb
    print("✅ Generic numba functions accessible")

    print("\n🎉 VectorBT Pro installation verified successfully!")
    print("⚠️  Note: Full auto-import disabled due to missing 'data' module in source")
    print("   Core backtesting functionality (portfolio, signals) is available")

except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    exit(1)
