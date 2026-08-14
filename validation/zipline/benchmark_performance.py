#!/usr/bin/env python3
"""Route the former Zipline benchmark entry point to retained performance evidence."""

from __future__ import annotations

import sys
from pathlib import Path

VALIDATION_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(VALIDATION_DIR))

from cross_framework_performance import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
