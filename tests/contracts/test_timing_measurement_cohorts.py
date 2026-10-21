"""Retained reference timings must keep their original identities and dates."""

import copy
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT / "validation"))

import generate_parity_claims as claims  # noqa: E402
import real_strategy_benchmark as benchmark  # noqa: E402


def _captured_report():
    return json.loads((_ROOT / "tests/fixtures/performance_reuse/metadata.json").read_text())


def test_actual_backtest_only_refresh_retains_valid_reference_and_pair_lineage():
    assert benchmark.measurement_cohort_failures(_captured_report()) == []


@pytest.mark.parametrize("fault", ["identity", "interval", "naive_time", "environment", "adapter"])
def test_reuse_validator_rejects_producer_metadata_that_changes_the_reference_or_execution_window(
    fault,
):
    report = _captured_report()
    cohorts = report["provenance"]["measurement_cohorts"]
    if fault == "identity":
        report["records"][0]["measurement_lineage"]["backtest_source_head"] = "0" * 40
    elif fault == "interval":
        cohorts["backtest"]["completed_at"] = cohorts["backtest"]["started_at"]
    elif fault == "naive_time":
        cohorts["framework"]["generated_at"] = "2026-10-13T23:57:58"
    elif fault == "environment":
        cohorts["framework"]["environment"]["frameworks"]["zipline"]["immutable_id"] = (
            "sha256:" + "0" * 64
        )
    else:
        cohorts["framework"]["provenance"]["sources"]["zipline"] = "0" * 64
    assert benchmark.measurement_cohort_failures(report)


def test_a_performance_review_cannot_keep_its_measurement_identity_after_changing_cohort_provenance():
    report = _captured_report()
    changed = copy.deepcopy(report)
    changed["provenance"]["measurement_cohorts"]["backtest"]["source_head"] = "0" * 40
    assert benchmark.measurement_digest(changed) != benchmark.measurement_digest(report)


def test_generated_caption_does_not_redate_reference_samples_when_the_report_is_written_later():
    report = _captured_report()
    report["generated_at"] = "2026-10-22T00:00:00+00:00"
    assert claims.measurement_caption(report) == (
        "Backtest measured 2026-10-21; reference measurements dated "
        "2026-10-13 were retained unchanged"
    )


def test_original_single_cohort_reports_keep_their_existing_caption_and_validation_contract():
    report = _captured_report()
    del report["provenance"]["measurement_cohorts"]
    assert benchmark.measurement_cohort_failures(report) == []
    assert claims.measurement_caption(report) == "Measured 2026-10-21"
