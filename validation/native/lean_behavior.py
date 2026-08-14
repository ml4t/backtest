#!/usr/bin/env python3
"""Run framework-native behavior cases against the frozen LEAN engine image."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

IMAGE = "quantconnect/lean@sha256:ecd62b0e418d40d1d7c0cd95e90a94e397642a21d2c8810614830c8a4e9a8f70"
PLATFORM_ARTIFACT = (
    "linux/amd64@sha256:cbe3f26b3f16c57be836b2cf913253d434f58e010c84ed038360d26b9df88307"
)
ENGINE_VERSION = "18001"
ENGINE_COMMIT = "278fcb3d1b815b63ccadba68d7ae54422e34b792"
CLI_VERSION = "1.0.228"
CLI_SHA256 = "eaa4c08f16295b76f005e429d9ca0d0453784dc1a40c1f5cbe5e50c02a05bd7c"

ROOT = Path(__file__).parents[2]
PROJECT_SOURCE = Path(__file__).parent / "lean_project"
SUPPORT_SOURCE = ROOT / "validation/lean/support"
CASES = (
    "timing",
    "default_models",
    "target_sizing",
    "submission_sequence",
    "buying_power_allowed",
    "buying_power_rejected",
    "buying_power_sequence",
    "explicit_costs",
    "default_full_fill",
    "fill_forward",
    "terminal_holding",
    "final_bar_order",
    "liquidation",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_equity(data_root: Path, ticker: str, *, missing_second_bar: bool = False) -> None:
    equity_root = data_root / "equity/usa"
    daily = equity_root / "daily"
    maps = equity_root / "map_files"
    factors = equity_root / "factor_files"
    for directory in (daily, maps, factors):
        directory.mkdir(parents=True, exist_ok=True)

    rows = [
        ("20240102", 100.0, 102.0, 99.0, 101.0, 100),
        ("20240103", 110.0, 112.0, 109.0, 111.0, 100),
        ("20240104", 120.0, 122.0, 119.0, 121.0, 100),
        ("20240105", 130.0, 132.0, 129.0, 131.0, 100),
        ("20240108", 140.0, 142.0, 139.0, 141.0, 100),
        ("20240109", 150.0, 152.0, 149.0, 151.0, 100),
        ("20240110", 160.0, 162.0, 159.0, 161.0, 100),
        ("20240111", 170.0, 172.0, 169.0, 171.0, 100),
    ]
    if missing_second_bar:
        rows.pop(1)
    lines = []
    for date, open_, high, low, close, volume in rows:
        prices = [int(round(value * 10_000)) for value in (open_, high, low, close)]
        lines.append(f"{date} 00:00,{prices[0]},{prices[1]},{prices[2]},{prices[3]},{volume}")

    ticker_lower = ticker.lower()
    with zipfile.ZipFile(
        daily / f"{ticker_lower}.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr(f"{ticker_lower}.csv", "\n".join(lines))
    (maps / f"{ticker_lower}.csv").write_text(
        f"20240102,{ticker_lower}\n20501231,{ticker_lower}\n", encoding="utf-8"
    )
    (factors / f"{ticker_lower}.csv").write_text(
        "20240102,1,1,1\n20501231,1,1,0\n", encoding="utf-8"
    )


def _prepare(root: Path) -> tuple[Path, Path]:
    project = root / "lean_project"
    shutil.copytree(PROJECT_SOURCE, project)
    shutil.copy2(SUPPORT_SOURCE / "lean.json", root / "lean.json")
    data = root / "data"
    shutil.copytree(SUPPORT_SOURCE / "data", data)
    _write_equity(data, "MLNATV")
    _write_equity(data, "MLMISS", missing_second_bar=True)
    return project, root / "lean.json"


def _command_version(command: Path) -> str:
    result = subprocess.run(
        [str(command), "--version"], check=True, capture_output=True, text=True, timeout=60
    )
    return result.stdout.strip()


def _image_identity() -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "image", "inspect", IMAGE],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(result.stdout)[0]
    labels = payload.get("Config", {}).get("Labels", {}) or {}
    return {
        "engine_version": labels.get("lean_version"),
        "image": IMAGE,
        "image_id": payload["Id"],
        "platform_artifact": PLATFORM_ARTIFACT,
        "python_version": labels.get("python_version"),
        "target_framework": labels.get("target_framework"),
    }


def _run_case(
    command: Path,
    root: Path,
    project: Path,
    lean_config: Path,
    case: str,
) -> dict[str, Any]:
    output = root / "outputs" / case
    result = subprocess.run(
        [
            str(command),
            "backtest",
            str(project),
            "--lean-config",
            str(lean_config),
            "--image",
            IMAGE,
            "--no-update",
            "--output",
            str(output),
            "--parameter",
            "case",
            case,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        error = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}".strip()
        raise RuntimeError(f"LEAN case {case} failed: {error[-20_000:]}")
    artifact = project / f"lean_native_{case}.json"
    if not artifact.is_file():
        raise RuntimeError(f"LEAN case {case} did not emit {artifact.name}")
    return json.loads(artifact.read_text(encoding="utf-8"))


def _equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, float) and isinstance(expected, (float, int)):
        return math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=1e-8)
    if isinstance(actual, list) and isinstance(expected, list) and len(actual) == len(expected):
        return all(_equal(left, right) for left, right in zip(actual, expected, strict=True))
    if isinstance(actual, dict) and isinstance(expected, dict) and actual.keys() == expected.keys():
        return all(_equal(actual[key], expected[key]) for key in actual)
    return bool(actual == expected)


def _project_contract(case: str, payload: dict[str, Any]) -> dict[str, Any]:
    events = [
        [
            event["status"],
            event["direction"],
            event["fill_quantity"],
            event["fill_price"],
            event["fee"],
            event["event_time_utc"],
        ]
        for event in payload["events"]
    ]
    final = {
        "cash": payload["cash"],
        "position": payload["position"],
        "total_fees": payload["total_fees"],
        "total_portfolio_value": payload["total_portfolio_value"],
    }
    if case == "default_models":
        return {"models": payload["models"]}
    if case == "target_sizing":
        return {"events": events, "final": final, "target_quantity": payload["target_quantity"]}
    if case == "explicit_costs":
        return {"events": events, "final": final, "models": payload["models"]}
    if case == "default_full_fill":
        return {
            "events": events,
            "final": final,
            "source_volume": payload["observations"][0]["volume"],
        }
    if case == "fill_forward":
        synthetic = next(row for row in payload["observations"] if row["date"] == "2024-01-03")
        return {"events": events, "final": final, "synthetic_bar": synthetic}
    return {"events": events, "final": final}


EXPECTED = {
    "timing": {
        "events": [
            ["SUBMITTED", "BUY", 0.0, 0.0, 0.0, "2024-01-02 21:00:00"],
            ["FILLED", "BUY", 1.0, 110.0, 1.0, "2024-01-03 21:00:00"],
        ],
        "final": {
            "cash": 9889.0,
            "position": 1.0,
            "total_fees": 1.0,
            "total_portfolio_value": 10060.0,
        },
    },
    "default_models": {
        "models": {
            "brokerage": "DefaultBrokerageModel",
            "buying_power": "SecurityMarginModel",
            "fee": "InteractiveBrokersFeeModel",
            "fill": "EquityFillModel",
            "leverage": 2.0,
            "slippage": "NullSlippageModel",
        }
    },
    "target_sizing": {
        "events": [
            ["SUBMITTED", "BUY", 0.0, 0.0, 0.0, "2024-01-02 21:00:00"],
            ["FILLED", "BUY", 98.0, 110.0, 1.0, "2024-01-03 21:00:00"],
        ],
        "final": {
            "cash": -781.0,
            "position": 98.0,
            "total_fees": 1.0,
            "total_portfolio_value": 15977.0,
        },
        "target_quantity": 98.0,
    },
    "submission_sequence": {
        "events": [
            ["SUBMITTED", "BUY", 0.0, 0.0, 0.0, "2024-01-02 21:00:00"],
            ["SUBMITTED", "SELL", 0.0, 0.0, 0.0, "2024-01-02 21:00:00"],
            ["FILLED", "BUY", 1.0, 110.0, 1.0, "2024-01-03 21:00:00"],
            ["FILLED", "SELL", -1.0, 110.0, 1.0, "2024-01-03 21:00:00"],
        ],
        "final": {
            "cash": 9998.0,
            "position": 0.0,
            "total_fees": 2.0,
            "total_portfolio_value": 9998.0,
        },
    },
    "buying_power_allowed": {
        "events": [
            ["SUBMITTED", "BUY", 0.0, 0.0, 0.0, "2024-01-02 21:00:00"],
            ["FILLED", "BUY", 150.0, 110.0, 1.0, "2024-01-03 21:00:00"],
        ],
        "final": {
            "cash": -6501.0,
            "position": 150.0,
            "total_fees": 1.0,
            "total_portfolio_value": 19149.0,
        },
    },
    "buying_power_rejected": {
        "events": [["INVALID", "BUY", 0.0, 0.0, 0.0, "2024-01-02 21:00:00"]],
        "final": {
            "cash": 10000.0,
            "position": 0.0,
            "total_fees": 0.0,
            "total_portfolio_value": 10000.0,
        },
    },
    "buying_power_sequence": {
        "events": [
            ["SUBMITTED", "BUY", 0.0, 0.0, 0.0, "2024-01-02 21:00:00"],
            ["SUBMITTED", "BUY", 0.0, 0.0, 0.0, "2024-01-02 21:00:00"],
            ["FILLED", "BUY", 150.0, 110.0, 1.0, "2024-01-03 21:00:00"],
            ["INVALID", "BUY", 0.0, 0.0, 0.0, "2024-01-03 21:00:00"],
        ],
        "final": {
            "cash": -6501.0,
            "position": 150.0,
            "total_fees": 1.0,
            "total_portfolio_value": 19149.0,
        },
    },
    "explicit_costs": {
        "events": [
            ["SUBMITTED", "BUY", 0.0, 0.0, 0.0, "2024-01-02 21:00:00"],
            ["FILLED", "BUY", 1.0, 110.111, 1.0, "2024-01-03 21:00:00"],
        ],
        "final": {
            "cash": 9888.889,
            "position": 1.0,
            "total_fees": 1.0,
            "total_portfolio_value": 10059.889,
        },
        "models": {
            "brokerage": "DefaultBrokerageModel",
            "buying_power": "SecurityMarginModel",
            "fee": "ConstantFeeModel",
            "fill": "EquityFillModel",
            "leverage": 2.0,
            "slippage": "ConstantSlippageModel",
        },
    },
    "default_full_fill": {
        "events": [
            ["SUBMITTED", "BUY", 0.0, 0.0, 0.0, "2024-01-02 21:00:00"],
            ["FILLED", "BUY", 150.0, 110.0, 1.0, "2024-01-03 21:00:00"],
        ],
        "final": {
            "cash": -6501.0,
            "position": 150.0,
            "total_fees": 1.0,
            "total_portfolio_value": 19149.0,
        },
        "source_volume": 100.0,
    },
    "fill_forward": {
        "events": [
            ["SUBMITTED", "BUY", 0.0, 0.0, 0.0, "2024-01-03 21:00:00"],
            ["FILLED", "BUY", 1.0, 120.0, 1.0, "2024-01-04 21:00:00"],
        ],
        "final": {
            "cash": 9879.0,
            "position": 1.0,
            "total_fees": 1.0,
            "total_portfolio_value": 10050.0,
        },
        "synthetic_bar": {
            "close": 101.0,
            "date": "2024-01-03",
            "fill_forward": True,
            "open": 100.0,
            "volume": 0.0,
        },
    },
    "terminal_holding": {
        "events": [
            ["SUBMITTED", "BUY", 0.0, 0.0, 0.0, "2024-01-02 21:00:00"],
            ["FILLED", "BUY", 1.0, 110.0, 1.0, "2024-01-03 21:00:00"],
        ],
        "final": {
            "cash": 9889.0,
            "position": 1.0,
            "total_fees": 1.0,
            "total_portfolio_value": 10060.0,
        },
    },
    "final_bar_order": {
        "events": [["SUBMITTED", "BUY", 0.0, 0.0, 0.0, "2024-01-11 21:00:00"]],
        "final": {
            "cash": 10000.0,
            "position": 0.0,
            "total_fees": 0.0,
            "total_portfolio_value": 10000.0,
        },
    },
    "liquidation": {
        "events": [
            ["SUBMITTED", "BUY", 0.0, 0.0, 0.0, "2024-01-02 21:00:00"],
            ["FILLED", "BUY", 1.0, 110.0, 1.0, "2024-01-03 21:00:00"],
            ["SUBMITTED", "SELL", 0.0, 0.0, 0.0, "2024-01-10 21:00:00"],
            ["FILLED", "SELL", -1.0, 170.0, 1.0, "2024-01-11 21:00:00"],
        ],
        "final": {
            "cash": 10058.0,
            "position": 0.0,
            "total_fees": 2.0,
            "total_portfolio_value": 10058.0,
        },
    },
}


def run(command: Path) -> dict[str, Any]:
    command = command.resolve()
    version = _command_version(command)
    if version != f"lean {CLI_VERSION}":
        raise RuntimeError(f"LEAN CLI differs: {version} != lean {CLI_VERSION}")
    image = _image_identity()
    if image["engine_version"] != ENGINE_VERSION:
        raise RuntimeError(
            f"LEAN engine version differs: {image['engine_version']} != {ENGINE_VERSION}"
        )

    with tempfile.TemporaryDirectory(prefix="ml4t-lean-native-") as temporary:
        temporary_root = Path(temporary)
        project, lean_config = _prepare(temporary_root)
        actual = {
            case: _run_case(command, temporary_root, project, lean_config, case) for case in CASES
        }

    checks = [
        {
            "id": case,
            "actual": actual[case],
            "contract_actual": _project_contract(case, actual[case]),
            "expected": EXPECTED[case],
            "passed": _equal(_project_contract(case, actual[case]), EXPECTED[case]),
        }
        for case in CASES
    ]
    project_files = {
        path.relative_to(PROJECT_SOURCE).as_posix(): _digest(path)
        for path in sorted(PROJECT_SOURCE.rglob("*"))
        if path.is_file()
    }
    support_files = {
        path.relative_to(SUPPORT_SOURCE).as_posix(): _digest(path)
        for path in sorted(SUPPORT_SOURCE.rglob("*"))
        if path.is_file()
    }
    data_payload = {
        "full": [100.0, 101.0, 110.0, 111.0, 120.0, 121.0, 170.0, 171.0],
        "missing_session": "MLMISS omits 2024-01-03 and enables fill_forward",
    }
    return {
        "schema_version": 1,
        "framework": "lean",
        "engine": {
            **image,
            "source_commit": ENGINE_COMMIT,
        },
        "cli": {
            "package": "lean",
            "version": CLI_VERSION,
            "immutable_id": f"sha256:{CLI_SHA256}",
        },
        "models": {
            "account_type": "Margin",
            "brokerage": "DefaultBrokerageModel",
            "security": "Equity USA, daily, adjusted normalization, leverage 2",
        },
        "source_references": {
            "algorithm_settings": (
                "https://github.com/QuantConnect/Lean/blob/"
                f"{ENGINE_COMMIT}/Common/AlgorithmSettings.cs"
            ),
            "brokerage": (
                "https://github.com/QuantConnect/Lean/blob/"
                f"{ENGINE_COMMIT}/Common/Brokerages/DefaultBrokerageModel.cs"
            ),
            "fee": (
                "https://github.com/QuantConnect/Lean/blob/"
                f"{ENGINE_COMMIT}/Common/Orders/Fees/InteractiveBrokersFeeModel.cs"
            ),
            "fill": (
                "https://github.com/QuantConnect/Lean/blob/"
                f"{ENGINE_COMMIT}/Common/Orders/Fills/EquityFillModel.cs"
            ),
            "margin": (
                "https://github.com/QuantConnect/Lean/blob/"
                f"{ENGINE_COMMIT}/Common/Securities/SecurityMarginModel.cs"
            ),
            "slippage": (
                "https://github.com/QuantConnect/Lean/blob/"
                f"{ENGINE_COMMIT}/Common/Orders/Slippage/ConstantSlippageModel.cs"
            ),
        },
        "data": {
            "format": "LEAN local equity daily zip, raw prices scaled by 10000",
            "payload_sha256": hashlib.sha256(
                json.dumps(data_payload, sort_keys=True).encode()
            ).hexdigest(),
            "sessions": [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
                "2024-01-09",
                "2024-01-10",
                "2024-01-11",
            ],
        },
        "oracle_files": project_files,
        "support_files": support_files,
        "oracle_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean-command", type=Path, default=ROOT / ".venv-lean/bin/lean")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        evidence = run(args.lean_command)
    except (FileNotFoundError, json.JSONDecodeError, OSError, RuntimeError, ValueError) as error:
        print(f"LEAN native behavior run failed: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failed = [check["id"] for check in evidence["checks"] if not check["passed"]]
    if failed:
        print(f"LEAN native behavior differs: {failed}", file=sys.stderr)
        return 1
    print(f"LEAN native behavior passed: {len(evidence['checks'])} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
