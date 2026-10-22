"""Retain the verified Data snapshot and the tables actually selected by Backtest."""

from __future__ import annotations

import json
from hashlib import sha256
from io import BytesIO
from typing import Any

import polars as pl


def frame_digest(frame: pl.DataFrame) -> str:
    buffer = BytesIO()
    frame.rechunk().write_ipc(buffer, compression="uncompressed")
    return sha256(buffer.getbuffer()).hexdigest()


def provenance_digest(value: dict[str, Any]) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_input_provenance(value: dict[str, Any]) -> None:
    if not isinstance(value, dict) or value.get("source") != "ml4t.data.snapshot":
        raise ValueError("Input provenance requires a verified Data snapshot receipt")
    for name in ("snapshot_id", "manifest_sha256", "manifest_file_sha256"):
        digest = value.get(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Input provenance requires SHA256 source identities")
    manifest = value.get("manifest")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("content"), dict):
        raise ValueError("Input provenance has no retained source manifest")
    if (
        manifest.get("snapshot_id") != value["snapshot_id"]
        or provenance_digest(manifest["content"]) != value["snapshot_id"]
    ):
        raise ValueError("Input provenance snapshot identity does not match its source manifest")
    expected = provenance_digest(
        {key: item for key, item in manifest.items() if key != "manifest_sha256"}
    )
    if (
        manifest.get("manifest_sha256") != value["manifest_sha256"]
        or expected != value["manifest_sha256"]
    ):
        raise ValueError("Input provenance source manifest checksum does not match")
    consumed = value.get("consumed_tables")
    tables = manifest["content"].get("tables")
    if (
        not isinstance(consumed, dict)
        or "prices" not in consumed
        or not isinstance(tables, dict)
        or not set(consumed)
        <= {"prices", "signals", "context", "corporate_actions", "capital_flows"}
        or any(not isinstance(name, str) or name not in tables for name in consumed.values())
    ):
        raise ValueError("Input provenance contains unknown consumed tables")
    provenance_digest(value)
    hashes = value.get("normalized_tables_sha256")
    if (
        value.get("normalized_table_format") != "arrow-ipc-rechunked-v1"
        or not isinstance(value.get("polars_version"), str)
        or not isinstance(hashes, dict)
        or set(hashes) != set(consumed)
        or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            for digest in hashes.values()
        )
    ):
        raise ValueError("Input provenance has no complete normalized table identities")
