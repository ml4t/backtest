"""Validate distribution contents and reproducibility."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import zipfile
from pathlib import Path

_ROOT = Path(__file__).parents[1]
_PACKAGE = _ROOT / "src" / "ml4t" / "backtest"
_TESTS = _ROOT / "tests"
_INTERNAL_NAMES = {"AGENTS.md", "CLAUDE.md"}


def _python_files(root: Path) -> set[str]:
    return {
        path.relative_to(_ROOT).as_posix()
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    }


def _single(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern} in {directory}, found {len(matches)}")
    return matches[0]


def _wheel_manifest(wheel: Path) -> tuple[set[str], str]:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
    if len(metadata) != 1:
        raise ValueError(f"Expected one wheel METADATA file, found {metadata}")
    dist_info = metadata[0].removesuffix("/METADATA")
    return names, dist_info


def _sdist_manifest(sdist: Path) -> set[str]:
    with tarfile.open(sdist) as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]
    roots = {name.split("/", 1)[0] for name in names}
    if len(roots) != 1:
        raise ValueError(f"Source distribution has unexpected roots: {sorted(roots)}")
    root = roots.pop()
    return {name.removeprefix(f"{root}/") for name in names}


def _manifest_diff(actual: set[str], expected: set[str], label: str) -> list[str]:
    failures: list[str] = []
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unexpected:
        failures.append(f"{label} has unexpected files: {unexpected}")
    if missing:
        failures.append(f"{label} is missing files: {missing}")
    return failures


def artifact_failures(directory: Path) -> list[str]:
    wheel = _single(directory, "*.whl")
    sdist = _single(directory, "*.tar.gz")
    wheel_files, dist_info = _wheel_manifest(wheel)
    sdist_files = _sdist_manifest(sdist)

    package_files = {path.removeprefix("src/") for path in _python_files(_PACKAGE)} | {
        "ml4t/backtest/py.typed"
    }
    expected_wheel = package_files | {
        f"{dist_info}/METADATA",
        f"{dist_info}/WHEEL",
        f"{dist_info}/licenses/LICENSE",
        f"{dist_info}/RECORD",
    }
    expected_sdist = (
        _python_files(_PACKAGE)
        | _python_files(_TESTS)
        | {
            ".gitignore",
            "CHANGELOG.md",
            "LICENSE",
            "README.md",
            "PKG-INFO",
            "pyproject.toml",
            "src/ml4t/backtest/py.typed",
        }
    )

    failures = _manifest_diff(wheel_files, expected_wheel, "wheel")
    failures.extend(_manifest_diff(sdist_files, expected_sdist, "sdist"))
    for name in wheel_files | sdist_files:
        path = Path(name)
        if _INTERNAL_NAMES.intersection(path.parts) or any(
            part in {".claude", ".workspace"} for part in path.parts
        ):
            failures.append(f"Distribution contains internal agent material: {name}")
    return failures


def reproducibility_failures(first: Path, second: Path) -> list[str]:
    failures: list[str] = []
    for pattern in ("*.whl", "*.tar.gz"):
        left = _single(first, pattern)
        right = _single(second, pattern)
        if left.name != right.name:
            failures.append(f"Artifact names differ: {left.name} != {right.name}")
            continue
        left_digest = hashlib.sha256(left.read_bytes()).hexdigest()
        right_digest = hashlib.sha256(right.read_bytes()).hexdigest()
        if left_digest != right_digest:
            failures.append(f"Artifact is not reproducible: {left.name}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_directory", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    failures = artifact_failures(args.artifact_directory)
    if args.compare is not None:
        failures.extend(artifact_failures(args.compare))
        failures.extend(reproducibility_failures(args.artifact_directory, args.compare))
    if failures:
        for failure in failures:
            print(failure)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
