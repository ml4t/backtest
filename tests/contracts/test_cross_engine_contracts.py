from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "validation" / "run_all_correctness.py"

FRAMEWORK_VENVS = {
    "vectorbt_oss": ".venv",
    "backtrader": ".venv-backtrader",
    "zipline": ".venv-zipline",
}


@pytest.mark.requires_comparison
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("framework", ["vectorbt_oss", "backtrader", "zipline"])
def test_cross_engine_scenario_01_contract(framework: str, tmp_path: Path) -> None:
    venv_dir = PROJECT_ROOT / FRAMEWORK_VENVS[framework]
    if not (venv_dir / "bin" / "python").exists():
        pytest.skip(f"{framework} environment not available: {venv_dir}")

    output_file = tmp_path / f"correctness_{framework}.md"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--framework",
        framework,
        "--scenarios",
        "01",
        "--output",
        str(output_file),
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    combined = f"{result.stdout}\n{result.stderr}"

    assert result.returncode == 0, combined
    assert "PASS" in combined and "FAIL" not in combined, combined
