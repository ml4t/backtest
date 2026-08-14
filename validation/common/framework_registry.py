"""Validated framework targets and comparison-claim inventory."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALIDATION_DIR = Path(__file__).parents[1]
DEFAULT_MANIFEST_PATH = VALIDATION_DIR / "framework_targets.toml"
DEFAULT_CLAIM_INVENTORY_PATH = VALIDATION_DIR / "claim_inventory.toml"
_EXACT_VERSION = re.compile(r"^[0-9][0-9A-Za-z._+-]*$")
_IMMUTABLE_ID = re.compile(r"^(?:git:[0-9a-f]{7,40}|sha256:[0-9a-f]{64})$")
_CLAIM_TYPES = {"behavioral", "correctness", "performance", "qualitative"}
_ACCESS_TYPES = {"public", "licensed"}


class ManifestError(ValueError):
    """A framework or claim manifest violates its schema."""


@dataclass(frozen=True)
class FrameworkTarget:
    """One immutable comparison target."""

    framework_id: str
    display_name: str
    package: str
    version: str
    profile: str
    python: str
    license: str
    access: str
    source: str
    immutable_id: str
    scenario_matrix: bool
    source_commit: str | None = None
    artifact: str | None = None
    environment: str | None = None
    python_env_var: str | None = None
    platform_artifact: str | None = None
    cli_version: str | None = None
    cli_immutable_id: str | None = None

    def evidence_metadata(self) -> dict[str, object]:
        """Return complete target identity for retained evidence."""
        metadata: dict[str, object] = {
            "display_name": self.display_name,
            "profile": self.profile,
            "package": self.package,
            "version": self.version,
            "python": self.python,
            "license": self.license,
            "access": self.access,
            "source": self.source,
            "immutable_id": self.immutable_id,
        }
        optional = {
            "commit": self.source_commit,
            "artifact": self.artifact,
            "platform_artifact": self.platform_artifact,
            "cli_version": self.cli_version,
            "cli_immutable_id": self.cli_immutable_id,
        }
        metadata.update({key: value for key, value in optional.items() if value is not None})
        return metadata


@dataclass(frozen=True)
class FrameworkManifest:
    """Frozen comparison target set."""

    schema_version: int
    freeze_date: str
    scenario_framework_ids: tuple[str, ...]
    targets: dict[str, FrameworkTarget]


@dataclass(frozen=True)
class Claim:
    """One externally visible comparison claim surface."""

    claim_id: str
    repository: str
    paths: tuple[str, ...]
    claim_type: str
    frameworks: tuple[str, ...]
    required_evidence: tuple[str, ...]
    status: str
    issue: str


@dataclass(frozen=True)
class UnresolvedFinding:
    """One known audit finding routed to an implementation issue."""

    finding_id: str
    finding: str
    issue: str


@dataclass(frozen=True)
class ClaimInventory:
    """Dated inventory of claims and unresolved findings."""

    schema_version: int
    audit_date: str
    claims: tuple[Claim, ...]
    unresolved: tuple[UnresolvedFinding, ...]


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as file:
            payload = tomllib.load(file)
    except tomllib.TOMLDecodeError as error:
        raise ManifestError(f"Invalid TOML in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ManifestError(f"Manifest must be a TOML table: {path}")
    return payload


def _required(table: dict[str, Any], field: str, *, context: str) -> Any:
    if field not in table:
        raise ManifestError(f"{context} missing required field '{field}'")
    return table[field]


def _string(table: dict[str, Any], field: str, *, context: str) -> str:
    value = _required(table, field, context=context)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{context} field '{field}' must be a non-empty string")
    return value


def _strings(table: dict[str, Any], field: str, *, context: str) -> tuple[str, ...]:
    value = _required(table, field, context=context)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ManifestError(f"{context} field '{field}' must be a non-empty string array")
    return tuple(value)


def load_framework_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> FrameworkManifest:
    """Load and validate frozen comparison targets."""
    payload = _load_toml(path)
    metadata = payload.get("metadata")
    frameworks = payload.get("framework")
    if not isinstance(metadata, dict) or not isinstance(frameworks, dict) or not frameworks:
        raise ManifestError("Framework manifest requires metadata and framework tables")

    schema_version = _required(metadata, "schema_version", context="metadata")
    if schema_version != 1:
        raise ManifestError(f"Unsupported framework manifest schema: {schema_version}")
    freeze_date = _string(metadata, "freeze_date", context="metadata")
    scenario_frameworks = _strings(metadata, "scenario_frameworks", context="metadata")

    targets: dict[str, FrameworkTarget] = {}
    for framework_id, raw_target in frameworks.items():
        context = f"framework.{framework_id}"
        if not isinstance(raw_target, dict):
            raise ManifestError(f"{context} must be a table")
        version = _string(raw_target, "version", context=context)
        if not _EXACT_VERSION.fullmatch(version):
            raise ManifestError(f"{context} version must be an exact version: {version!r}")
        immutable_id = _string(raw_target, "immutable_id", context=context)
        if not _IMMUTABLE_ID.fullmatch(immutable_id):
            raise ManifestError(f"{context} immutable_id is not an immutable Git or SHA-256 ID")
        access = _string(raw_target, "access", context=context)
        if access not in _ACCESS_TYPES:
            raise ManifestError(f"{context} access must be one of {sorted(_ACCESS_TYPES)}")
        scenario_matrix = _required(raw_target, "scenario_matrix", context=context)
        if not isinstance(scenario_matrix, bool):
            raise ManifestError(f"{context} scenario_matrix must be a boolean")

        target = FrameworkTarget(
            framework_id=framework_id,
            display_name=_string(raw_target, "display_name", context=context),
            package=_string(raw_target, "package", context=context),
            version=version,
            profile=_string(raw_target, "profile", context=context),
            python=_string(raw_target, "python", context=context),
            license=_string(raw_target, "license", context=context),
            access=access,
            source=_string(raw_target, "source", context=context),
            immutable_id=immutable_id,
            scenario_matrix=scenario_matrix,
            source_commit=raw_target.get("source_commit"),
            artifact=raw_target.get("artifact"),
            environment=raw_target.get("environment"),
            python_env_var=raw_target.get("python_env_var"),
            platform_artifact=raw_target.get("platform_artifact"),
            cli_version=raw_target.get("cli_version"),
            cli_immutable_id=raw_target.get("cli_immutable_id"),
        )
        if scenario_matrix and (not target.environment or not target.python_env_var):
            raise ManifestError(f"{context} scenario target lacks interpreter configuration")
        targets[framework_id] = target

    if len(set(scenario_frameworks)) != len(scenario_frameworks):
        raise ManifestError("metadata scenario_frameworks contains duplicates")
    unknown = set(scenario_frameworks) - set(targets)
    if unknown:
        raise ManifestError(
            f"metadata scenario_frameworks names unknown targets: {sorted(unknown)}"
        )
    declared = {key for key, target in targets.items() if target.scenario_matrix}
    if set(scenario_frameworks) != declared:
        raise ManifestError("metadata scenario_frameworks disagrees with target declarations")

    return FrameworkManifest(schema_version, freeze_date, scenario_frameworks, targets)


def load_claim_inventory(
    path: Path = DEFAULT_CLAIM_INVENTORY_PATH,
    *,
    manifest: FrameworkManifest | None = None,
) -> ClaimInventory:
    """Load and validate the dated comparison-claim inventory."""
    target_manifest = manifest or load_framework_manifest()
    payload = _load_toml(path)
    metadata = payload.get("metadata")
    raw_claims = payload.get("claim")
    raw_unresolved = payload.get("unresolved")
    if not isinstance(metadata, dict) or not isinstance(raw_claims, list) or not raw_claims:
        raise ManifestError("Claim inventory requires metadata and claim tables")
    if not isinstance(raw_unresolved, list):
        raise ManifestError("Claim inventory requires unresolved tables")

    schema_version = _required(metadata, "schema_version", context="metadata")
    if schema_version != 1:
        raise ManifestError(f"Unsupported claim inventory schema: {schema_version}")
    audit_date = _string(metadata, "audit_date", context="metadata")

    claims: list[Claim] = []
    claim_ids: set[str] = set()
    for index, raw_claim in enumerate(raw_claims):
        context = f"claim[{index}]"
        if not isinstance(raw_claim, dict):
            raise ManifestError(f"{context} must be a table")
        claim_id = _string(raw_claim, "id", context=context)
        if claim_id in claim_ids:
            raise ManifestError(f"Duplicate claim id: {claim_id}")
        claim_ids.add(claim_id)
        claim_type = _string(raw_claim, "type", context=context)
        if claim_type not in _CLAIM_TYPES:
            raise ManifestError(f"{context} has unknown claim type: {claim_type}")
        frameworks = _strings(raw_claim, "frameworks", context=context)
        unknown = set(frameworks) - set(target_manifest.targets)
        if unknown:
            raise ManifestError(f"{context} names unknown framework targets: {sorted(unknown)}")
        claims.append(
            Claim(
                claim_id=claim_id,
                repository=_string(raw_claim, "repository", context=context),
                paths=_strings(raw_claim, "paths", context=context),
                claim_type=claim_type,
                frameworks=frameworks,
                required_evidence=_strings(raw_claim, "required_evidence", context=context),
                status=_string(raw_claim, "status", context=context),
                issue=_string(raw_claim, "issue", context=context),
            )
        )

    unresolved: list[UnresolvedFinding] = []
    finding_ids: set[str] = set()
    for index, raw_finding in enumerate(raw_unresolved):
        context = f"unresolved[{index}]"
        if not isinstance(raw_finding, dict):
            raise ManifestError(f"{context} must be a table")
        finding_id = _string(raw_finding, "id", context=context)
        if finding_id in finding_ids:
            raise ManifestError(f"Duplicate unresolved finding id: {finding_id}")
        finding_ids.add(finding_id)
        unresolved.append(
            UnresolvedFinding(
                finding_id=finding_id,
                finding=_string(raw_finding, "finding", context=context),
                issue=_string(raw_finding, "issue", context=context),
            )
        )

    return ClaimInventory(schema_version, audit_date, tuple(claims), tuple(unresolved))
