"""Create and verify commit-bound release-candidate evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import cast

_SCHEMA_VERSION = 1
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
REQUIRED_GATES = frozenset(
    {
        "compatibility",
        "correctness",
        "coverage",
        "documentation",
        "lint",
        "packaging",
        "parity",
        "performance",
        "security",
        "typecheck",
    }
)


def _object_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        return None
    return cast(dict[str, object], value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distribution_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
    )


def _metadata_bytes(path: Path) -> bytes:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(names) != 1:
                raise ValueError(f"Expected one METADATA file in {path.name}, found {names}")
            return archive.read(names[0])
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path) as archive:
            members = [
                member for member in archive.getmembers() if member.name.endswith("/PKG-INFO")
            ]
            if len(members) != 1:
                raise ValueError(f"Expected one PKG-INFO file in {path.name}")
            extracted = archive.extractfile(members[0])
            if extracted is None:
                raise ValueError(f"Could not read PKG-INFO from {path.name}")
            return extracted.read()
    raise ValueError(f"Unsupported distribution file: {path.name}")


def _package_identity(paths: list[Path]) -> tuple[str, str]:
    identities: set[tuple[str, str]] = set()
    for path in paths:
        metadata = BytesParser().parsebytes(_metadata_bytes(path))
        name = metadata.get("Name")
        version = metadata.get("Version")
        if not name or not version:
            raise ValueError(f"Distribution metadata lacks Name or Version: {path.name}")
        identities.add((name, version))
    if len(identities) != 1:
        raise ValueError(f"Distribution package identities differ: {sorted(identities)}")
    return identities.pop()


def gate_failures(gates: dict[str, str]) -> list[str]:
    failures: list[str] = []
    missing = sorted(REQUIRED_GATES - gates.keys())
    unexpected = sorted(gates.keys() - REQUIRED_GATES)
    if missing:
        failures.append(f"Candidate is missing release gates: {missing}")
    if unexpected:
        failures.append(f"Candidate has unexpected release gates: {unexpected}")
    failures.extend(
        f"Release gate did not pass: {name}={result}"
        for name, result in sorted(gates.items())
        if name in REQUIRED_GATES and result != "success"
    )
    return failures


def create_manifest(
    directory: Path,
    *,
    commit: str,
    repository: str,
    gates: dict[str, str],
) -> dict[str, object]:
    if _COMMIT.fullmatch(commit) is None:
        raise ValueError("Release candidate commit must be a full lowercase SHA")
    if _REPOSITORY.fullmatch(repository) is None:
        raise ValueError("Release candidate repository must have owner/name form")
    failures = gate_failures(gates)
    if failures:
        raise ValueError("; ".join(failures))
    paths = _distribution_paths(directory)
    if len(paths) != 2 or sum(path.suffix == ".whl" for path in paths) != 1:
        raise ValueError("Release candidate requires exactly one wheel and one source distribution")
    package_name, package_version = _package_identity(paths)
    return {
        "schema_version": _SCHEMA_VERSION,
        "source": {"repository": repository, "commit": commit},
        "package": {"name": package_name, "version": package_version},
        "gates": dict(sorted(gates.items())),
        "files": {
            path.name: {"sha256": _sha256(path), "size": path.stat().st_size} for path in paths
        },
    }


def candidate_failures(
    manifest: dict[str, object],
    directory: Path,
    *,
    expected_commit: str,
    expected_repository: str,
    expected_tag: str | None = None,
) -> list[str]:
    failures: list[str] = []
    if manifest.get("schema_version") != _SCHEMA_VERSION:
        failures.append(f"Unsupported candidate schema: {manifest.get('schema_version')!r}")

    source = _object_mapping(manifest.get("source"))
    if source is None:
        failures.append("Candidate source must be an object")
    else:
        if source.get("commit") != expected_commit:
            failures.append(
                f"Candidate commit is stale: {source.get('commit')!r} != {expected_commit!r}"
            )
        if source.get("repository") != expected_repository:
            failures.append(
                "Candidate repository differs: "
                f"{source.get('repository')!r} != {expected_repository!r}"
            )

    gates = manifest.get("gates")
    if not isinstance(gates, dict) or not all(
        isinstance(name, str) and isinstance(result, str) for name, result in gates.items()
    ):
        failures.append("Candidate gates must be a string mapping")
    else:
        failures.extend(gate_failures(cast(dict[str, str], gates)))

    package = _object_mapping(manifest.get("package"))
    if package is None:
        failures.append("Candidate package must be an object")
    elif expected_tag is not None and expected_tag != f"v{package.get('version')}":
        failures.append(
            f"Release tag does not match candidate version: {expected_tag!r} != "
            f"'v{package.get('version')}'"
        )

    files = _object_mapping(manifest.get("files"))
    if files is None:
        failures.append("Candidate files must be an object")
        return failures
    actual_paths = {path.name: path for path in _distribution_paths(directory)}
    declared_names = set(files)
    actual_names = set(actual_paths)
    missing = sorted(declared_names - actual_names)
    unexpected = sorted(actual_names - declared_names)
    if missing:
        failures.append(f"Candidate distributions are missing: {missing}")
    if unexpected:
        failures.append(f"Candidate has undeclared distributions: {unexpected}")
    for name in sorted(declared_names & actual_names):
        record = _object_mapping(files[name])
        if record is None:
            failures.append(f"Candidate file record must be an object: {name}")
            continue
        path = actual_paths[name]
        actual_digest = _sha256(path)
        if record.get("sha256") != actual_digest:
            failures.append(f"Candidate digest mismatch: {name}")
        if record.get("size") != path.stat().st_size:
            failures.append(f"Candidate size mismatch: {name}")
    try:
        paths = list(actual_paths.values())
        if len(paths) == 2:
            name, version = _package_identity(paths)
            if package is not None and package != {"name": name, "version": version}:
                failures.append("Candidate package identity differs from distribution metadata")
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
        failures.append(str(error))
    return failures


def index_failures(manifest: dict[str, object], payload: dict[str, object]) -> list[str]:
    files = _object_mapping(manifest.get("files"))
    package = _object_mapping(manifest.get("package"))
    if files is None or package is None:
        return ["Candidate manifest lacks package or file records"]
    info = _object_mapping(payload.get("info"))
    urls = payload.get("urls")
    if info is None or not isinstance(urls, list):
        return ["Package-index response lacks info or urls"]
    failures: list[str] = []
    if info.get("name") != package.get("name") or info.get("version") != package.get("version"):
        failures.append("Published package identity differs from the candidate")
    published: dict[str, str | None] = {}
    for raw_item in urls:
        item = _object_mapping(raw_item)
        if item is None or not isinstance(item.get("filename"), str):
            continue
        filename = cast(str, item["filename"])
        digests = _object_mapping(item.get("digests"))
        digest = digests.get("sha256") if digests is not None else None
        published[filename] = digest if isinstance(digest, str) else None
    if set(published) != set(files):
        failures.append(
            f"Published files differ from candidate: published={sorted(published)}, "
            f"candidate={sorted(files)}"
        )
    for name in sorted(set(published) & set(files)):
        record = _object_mapping(files[name])
        if record is None or published[name] != record.get("sha256"):
            failures.append(f"Published digest differs from candidate: {name}")
    return failures


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return cast(dict[str, object], payload)


def _parse_gates(values: list[str]) -> dict[str, str]:
    gates: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Gate must have name=result form: {value}")
        name, result = value.split("=", 1)
        if not name or name in gates:
            raise ValueError(f"Duplicate or empty gate: {name!r}")
        gates[name] = result
    return gates


def _index_payload(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("Package-index response root must be an object")
    return cast(dict[str, object], payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--dist", type=Path, required=True)
    create.add_argument("--commit", required=True)
    create.add_argument("--repository", required=True)
    create.add_argument("--gate", action="append", default=[])
    create.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--dist", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--expected-commit", required=True)
    verify.add_argument("--expected-repository", required=True)
    verify.add_argument("--expected-tag")

    verify_index = subparsers.add_parser("verify-index")
    verify_index.add_argument("--manifest", type=Path, required=True)
    verify_index.add_argument("--index-url", default="https://pypi.org/pypi")
    verify_index.add_argument("--attempts", type=int, default=12)
    verify_index.add_argument("--delay", type=float, default=5.0)
    args = parser.parse_args()

    try:
        if args.command == "create":
            manifest = create_manifest(
                args.dist,
                commit=args.commit,
                repository=args.repository,
                gates=_parse_gates(args.gate),
            )
            args.output.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            return 0
        manifest = _load_json(args.manifest)
        if args.command == "verify":
            failures = candidate_failures(
                manifest,
                args.dist,
                expected_commit=args.expected_commit,
                expected_repository=args.expected_repository,
                expected_tag=args.expected_tag,
            )
        else:
            package = _object_mapping(manifest.get("package"))
            if package is None:
                raise ValueError("Candidate manifest lacks package identity")
            url = (
                f"{args.index_url.rstrip('/')}/{package.get('name')}/{package.get('version')}/json"
            )
            failures = ["Package-index verification did not run"]
            for attempt in range(1, args.attempts + 1):
                try:
                    failures = index_failures(manifest, _index_payload(url))
                    if not failures:
                        break
                except (OSError, ValueError, urllib.error.URLError) as error:
                    failures = [f"Package-index verification failed: {error}"]
                if attempt < args.attempts:
                    time.sleep(args.delay)
        for failure in failures:
            print(failure)
        return int(bool(failures))
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        tarfile.TarError,
        zipfile.BadZipFile,
    ) as error:
        print(f"Release candidate evidence is invalid: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
