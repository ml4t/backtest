"""Contracts for the bounded current real-strategy inventory."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import polars as pl
import pytest


def _load_module():
    path = Path(__file__).parents[2] / "validation" / "real_strategy_corpus.py"
    spec = importlib.util.spec_from_file_location("ml4t_real_strategy_corpus", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _artifact_tree(tmp_path: Path, case_id: str = "etfs") -> tuple[Path, dict[str, str]]:
    artifact_root = tmp_path / "artifacts"
    run_log = artifact_root / "case_studies" / case_id / "run_log"
    backtest_hash = "a" * 12
    prediction_hash = "b" * 12
    backtest = run_log / "backtest" / backtest_hash
    prediction = run_log / "predictions" / prediction_hash
    backtest.mkdir(parents=True)
    prediction.mkdir(parents=True)
    (run_log / "registry.db").write_bytes(b"registry")
    pl.DataFrame({"timestamp": ["2024-01-01"], "symbol": ["SPY"], "y_score": [0.1]}).write_parquet(
        prediction / "predictions.parquet"
    )
    spec = {
        "backtest_config": {
            "metadata": {"prediction_hash": prediction_hash},
            "execution": {"execution_mode": "next_bar"},
            "commission": {"model": "none"},
            "slippage": {"model": "none"},
            "position_sizing": {"share_type": "integer"},
        },
        "strategy": {
            "signal": {"method": "equal_weight_top_k", "top_k": 1},
            "rebalance": {"mode": "engine", "cadence": "daily_close"},
        },
    }
    (backtest / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    for name in (
        "daily_returns.parquet",
        "equity.parquet",
        "fills.parquet",
        "portfolio_state.parquet",
        "trades.parquet",
        "weights.parquet",
    ):
        pl.DataFrame({"value": [1.0]}).write_parquet(backtest / name)
    return artifact_root, {
        "val_backtest_hash": backtest_hash,
        "val_prediction_hash": prediction_hash,
        "val_stage": "signal",
        "label": "fwd_ret_1d",
    }


def test_case_contract_contains_selected_real_case_studies() -> None:
    module = _load_module()

    cases = module.load_case_contract()

    assert len(cases) == 4
    assert {case["id"] for case in cases} == {
        "cme_futures",
        "crypto_perps_funding",
        "etfs",
        "us_equities_panel",
    }
    assert all(case["production_path"] == "event_driven" for case in cases)


def test_case_record_retains_predictions_spec_and_outputs(tmp_path: Path) -> None:
    module = _load_module()
    artifact_root, lineage = _artifact_tree(tmp_path)
    case = next(case for case in module.load_case_contract() if case["id"] == "etfs")

    record = module.build_case_record(case, lineage, artifact_root=artifact_root)

    assert record["case_study"] == "etfs"
    assert record["production_path"] == "event_driven"
    assert record["inputs"]["predictions.parquet"]["rows"] == 1
    assert record["backtest_artifacts"]["fills.parquet"]["rows"] == 1
    assert record["strategy"]["execution"] == {"execution_mode": "next_bar"}
    assert all("sha256" in value for value in record["backtest_artifacts"].values())


def test_case_record_fails_when_required_output_is_missing(tmp_path: Path) -> None:
    module = _load_module()
    artifact_root, lineage = _artifact_tree(tmp_path)
    case = next(case for case in module.load_case_contract() if case["id"] == "etfs")
    missing = (
        artifact_root
        / "case_studies"
        / "etfs"
        / "run_log"
        / "backtest"
        / lineage["val_backtest_hash"]
        / "fills.parquet"
    )
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="fills.parquet"):
        module.build_case_record(case, lineage, artifact_root=artifact_root)


def test_case_record_detects_prediction_mismatch(tmp_path: Path) -> None:
    module = _load_module()
    artifact_root, lineage = _artifact_tree(tmp_path)
    case = next(case for case in module.load_case_contract() if case["id"] == "etfs")
    spec_path = (
        artifact_root
        / "case_studies"
        / "etfs"
        / "run_log"
        / "backtest"
        / lineage["val_backtest_hash"]
        / "spec.json"
    )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["backtest_config"]["metadata"]["prediction_hash"] = "wrong"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(ValueError, match="does not name selected prediction"):
        module.build_case_record(case, lineage, artifact_root=artifact_root)
