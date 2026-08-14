#!/usr/bin/env python3
"""Compatibility entry point for the retained cross-framework performance harness."""

from __future__ import annotations

import sys
from pathlib import Path

VALIDATION_DIR = Path(__file__).parent
sys.path.insert(0, str(VALIDATION_DIR))

from common.framework_registry import load_framework_manifest  # noqa: E402
from cross_framework_performance import main as performance_main  # noqa: E402

FRAMEWORKS: dict[str, dict[str, str | None]] = {
    "ml4t": {"venv": ".venv", "display_name": "ml4t.backtest"}
}
FRAMEWORKS.update(
    {
        framework_id: {
            "venv": target.environment,
            "display_name": target.display_name,
        }
        for framework_id, target in load_framework_manifest().targets.items()
    }
)


if __name__ == "__main__":
    raise SystemExit(performance_main())
