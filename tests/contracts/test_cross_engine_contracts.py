from __future__ import annotations

import os
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

FRAMEWORK_PROFILES = {
    "vectorbt_oss": "vectorbt",
    "backtrader": "backtrader",
    "zipline": "zipline",
}


@pytest.mark.requires_comparison
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("framework", ["vectorbt_oss", "backtrader", "zipline"])
def test_cross_engine_scenario_01_contract(framework: str, tmp_path: Path) -> None:
    venv_dir = PROJECT_ROOT / FRAMEWORK_VENVS[framework]
    inproc = os.getenv("ML4T_COMPARISON_INPROC") == "1"

    if (venv_dir / "bin" / "python").exists() and not inproc:
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
    else:
        scenario_script = next(
            (PROJECT_ROOT / "validation" / framework).glob("scenario_01_*.py"), None
        )
        if scenario_script is None:
            pytest.skip(f"scenario_01 script missing for framework={framework}")
        env = {**os.environ, "ML4T_PROFILE": FRAMEWORK_PROFILES[framework]}
        result = subprocess.run(
            [sys.executable, str(scenario_script)],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        combined = f"{result.stdout}\n{result.stderr}"

    assert result.returncode == 0, combined
    assert "PASS" in combined and "FAIL" not in combined, combined
