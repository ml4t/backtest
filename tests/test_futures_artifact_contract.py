"""Actual Engine artifacts must retain the account method needed to read their dollars."""

import json
from dataclasses import replace

import polars as pl
import pytest

from ml4t.backtest import AssetClass, BacktestResult, ContractSpec
from ml4t.backtest.result import ArtifactError, ArtifactReadError, ArtifactWriteError
from tests.test_futures_variation import run_future


@pytest.mark.parametrize(
    "terms,message",
    [
        ({"accounting_method": "cash"}, "accounting_method"),
        ({"settlement_policy": "weekly"}, "settlement_policy"),
        ({"asset_class": AssetClass.EQUITY}, "FUTURE"),
        ({"currency": None}, "currency"),
    ],
)
def test_invalid_variation_declaration_cannot_construct_an_instrument(terms, message):
    valid = {"asset_class": AssetClass.FUTURE, "accounting_method": "futures_variation"}
    with pytest.raises(ValueError, match=message):
        ContractSpec("ES", **(valid | terms))


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("account_method", "signed_notional", "per-instrument"),
        ("schema_version", 5, "schema version 6"),
        ("instrument_accounting", None, "requires declared"),
        ("futures_content_sha256", "0" * 64, "content identity"),
    ],
)
def test_reader_rejects_method_and_identity_loss_from_a_real_variation_artifact(
    tmp_path,
    field,
    value,
    message,
):
    _, result, _ = run_future(1)
    result.to_parquet(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match=message):
        BacktestResult.from_parquet(tmp_path)


def test_missing_variation_table_cannot_be_recovered_as_a_complete_cash_account(tmp_path):
    _, result, _ = run_future(1)
    result.to_parquet(tmp_path)
    (tmp_path / "futures_settlements.parquet").unlink()
    with pytest.raises(ArtifactReadError, match="ledger is missing"):
        BacktestResult.from_parquet(tmp_path, recovery=True)


def test_account_export_cannot_drop_margin_observations_while_preserving_only_nav(tmp_path):
    _, result, _ = run_future(1)
    result.to_parquet(tmp_path)
    path = tmp_path / "portfolio_state.parquet"
    pl.read_parquet(path).drop("available_funds").write_parquet(path)
    with pytest.raises(ArtifactReadError, match="margin observations"):
        BacktestResult.from_parquet(tmp_path)


def test_variant_writer_refuses_omitted_or_misaligned_runtime_capture(tmp_path):
    _, result, _ = run_future(1)
    for field, value, message in [
        ("futures_settlements", None, "no captured"),
        ("accounting_observations", result.accounting_observations[:-1], "must match"),
        ("instrument_accounting", None, "requires instrument declarations"),
        ("account_method", "unknown", "Unknown account_method"),
        ("holdings_sampling", "unknown", "Unknown holdings_sampling"),
    ]:
        with pytest.raises(ArtifactError, match=message):
            replace(result, **{field: value}).to_parquet(tmp_path)
    result.to_parquet(tmp_path)
    assert (
        BacktestResult.from_parquet(tmp_path).to_futures_settlements_dataframe()["cash_delta"].sum()
        == 500
    )


def test_disabled_account_cannot_silently_acquire_an_orphan_futures_ledger(tmp_path):
    _, result, _ = run_future(1, legacy=True)
    with pytest.raises(ValueError, match="no captured"):
        result.to_futures_settlements_dataframe()
    with pytest.raises(ArtifactWriteError, match="require per-instrument declarations"):
        replace(result, futures_settlements=[]).to_parquet(tmp_path)


def test_accounting_columns_require_a_declared_method_in_legacy_artifacts(tmp_path):
    _, result, _ = run_future(1)
    result.to_parquet(tmp_path)
    path = tmp_path / "manifest.json"
    original = json.loads(path.read_text())
    legacy = {**original, "schema_version": 3, "account_method": "signed_notional"}
    legacy.pop("instrument_accounting")
    path.write_text(json.dumps(legacy))
    with pytest.raises(ArtifactError, match="Futures settlements require"):
        BacktestResult.from_parquet(tmp_path)
    legacy["components"].pop("futures_settlements")
    path.write_text(json.dumps(legacy))
    with pytest.raises(ArtifactReadError, match="Variation observations require"):
        BacktestResult.from_parquet(tmp_path)
