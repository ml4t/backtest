"""Regression test: ml4t-backtest[lean] reproduces LEAN on real case-study weights.

Uses the committed chapter16 LEAN workspaces (weights + daily zips + the LEAN-side
fills the algorithm logged). The ml4t[lean] reconstruction must match LEAN's fill
multiset and terminal value exactly. No Docker required — the LEAN side is the
committed reference artifact; only the ml4t side is recomputed here.
"""

from __future__ import annotations

import lzma
from pathlib import Path

import pytest

from ml4t.backtest._validation.case_study_lean import (
    _read_csv_fixture,
    compare,
    lean_side,
    run_ml4t_lean,
)

WORKSPACE = Path(__file__).resolve().parents[2] / "validation" / "lean" / "workspace"
DATA_DAILY = WORKSPACE / "data" / "equity" / "usa" / "daily"
CASE_STUDIES = [
    "chapter16_etfs",
    "chapter16_sp500_equity_option_analytics",
    "chapter16_us_equities_panel",
]


def _has_reference_artifacts(workspace_dir: Path) -> bool:
    return (
        (workspace_dir / "ml4t_order_events.csv").exists()
        or (workspace_dir / "ml4t_order_events.csv.xz").exists()
        or (workspace_dir / "ml4t_order_events.csv.gz").exists()
        or bool(list(workspace_dir.glob("ml4t_order_events.csv.part*.xz")))
    )


def test_retained_compressed_surface_precedes_ignored_runtime_file(tmp_path: Path) -> None:
    (tmp_path / "surface.csv").write_text("value\n1\n", encoding="utf-8")
    (tmp_path / "surface.csv.xz").write_bytes(lzma.compress(b"value\n2\n"))

    assert _read_csv_fixture(tmp_path, "surface.csv")["value"].tolist() == [2]


def test_lean_side_accepts_current_uppercase_order_status(tmp_path: Path) -> None:
    (tmp_path / "ml4t_symbol_map.json").write_text('{"AAAA": "AAPL"}', encoding="utf-8")
    (tmp_path / "ml4t_daily_equity.csv").write_text(
        "timestamp,equity\n2024-01-03,10010\n",
        encoding="utf-8",
    )
    (tmp_path / "ml4t_order_events.csv").write_text(
        "timestamp,symbol,status,direction,fill_quantity,fill_price\n"
        "2024-01-03,AAAA,FILLED,BUY,1,110\n",
        encoding="utf-8",
    )

    result = lean_side(tmp_path)

    assert result["n_trades"] == 1
    row = result["fills"].iloc[0]
    assert str(row["timestamp"].date()) == "2024-01-03"
    assert row["asset"] == "AAPL"
    assert row["side"] == "buy"
    assert row["quantity"] == 1.0
    assert row["price"] == 110.0


# The generic equity-decomposition invariant (initial + closed_pnl + open_pnl ==
# final_value) does not model 2x margin, but parity here is asserted directly
# against LEAN's terminal value (external ground truth), which matches exactly.
@pytest.mark.no_invariant_check
@pytest.mark.parametrize("project", CASE_STUDIES)
def test_case_study_lean_parity(project: str) -> None:
    workspace_dir = WORKSPACE / project
    assert _has_reference_artifacts(workspace_dir), (
        f"LEAN reference artifacts missing for {project}"
    )

    lean = lean_side(workspace_dir)
    ml4t = run_ml4t_lean(workspace_dir, DATA_DAILY)
    result = compare(lean, ml4t)

    assert result["sorted_fill_multiset_match"], (
        f"{project}: fill multiset diverges "
        f"(lean={result['lean_fills']} ml4t={result['ml4t_fills']})"
    )
    assert result["fill_gap"] == 0
    assert result["canonical_final_value_match"]
    assert result["canonical_final_value_gap_usd"] == 0.0
    assert abs(result["final_value_gap_usd"]) < 1e-4, (
        f"{project}: terminal value gap ${result['final_value_gap_usd']:.6f}"
    )
