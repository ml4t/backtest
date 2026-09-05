#!/usr/bin/env python3
"""Build and verify isolated, locked comparison-framework environments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

VALIDATION_DIR = Path(__file__).parent
PROJECT_ROOT = VALIDATION_DIR.parent
ENVIRONMENTS_DIR = VALIDATION_DIR / "environments"
sys.path.insert(0, str(VALIDATION_DIR))

from common.framework_registry import FrameworkTarget, load_framework_manifest  # noqa: E402

PUBLIC_FRAMEWORKS = ("vectorbt_oss", "backtrader", "zipline")
BUILDABLE_FRAMEWORKS = (*PUBLIC_FRAMEWORKS, "vectorbt_pro", "lean")


def project_directory(framework: str) -> Path:
    """Return the locked uv project for one framework."""
    return ENVIRONMENTS_DIR / framework


def environment_path(target: FrameworkTarget, root: Path = PROJECT_ROOT) -> Path:
    """Resolve a target environment below an explicit root."""
    if target.environment is None:
        raise ValueError(f"Framework has no environment path: {target.framework_id}")
    return root / target.environment


def definition_failures(framework: str, target: FrameworkTarget) -> list[str]:
    """Check that the lock input names the frozen target exactly."""
    project = project_directory(framework)
    failures: list[str] = []
    lock_path = project / "uv.lock"
    if not lock_path.is_file():
        failures.append(f"Missing lockfile: {lock_path}")
        lock_text = ""
    else:
        lock_text = lock_path.read_text(encoding="utf-8")
    try:
        with (project / "pyproject.toml").open("rb") as file:
            payload = tomllib.load(file)
        dependencies = payload["project"]["dependencies"]
    except (KeyError, OSError, tomllib.TOMLDecodeError) as error:
        return [f"Invalid environment definition for {framework}: {error}"]
    if not isinstance(dependencies, list) or not all(
        isinstance(dependency, str) for dependency in dependencies
    ):
        return [f"Environment dependencies must be a string array: {framework}"]
    normalized_package = target.package.lower().replace("_", "-")
    matching = [
        dependency
        for dependency in dependencies
        if dependency.split("@", 1)[0].split("=", 1)[0].strip().lower().replace("_", "-")
        == normalized_package
    ]
    if len(matching) != 1:
        failures.append(f"Environment must define one {target.package} dependency: {framework}")
    elif framework == "vectorbt_pro":
        if not matching[0].startswith(
            "vectorbtpro @ git+https://github.com/polakowo/vectorbt.pro.git@"
        ):
            failures.append("VectorBT Pro environment must use GitHub CLI authenticated HTTPS")
        if target.source_commit is None or f"@{target.source_commit}" not in matching[0]:
            failures.append("VectorBT Pro environment does not pin the manifest commit")
    else:
        expected_version = target.cli_version if framework == "lean" else target.version
        if expected_version is None or f"=={expected_version}" not in matching[0]:
            failures.append(f"Environment does not pin {target.package}=={expected_version}")
    expected_lock_id = target.cli_immutable_id if framework == "lean" else target.immutable_id
    if expected_lock_id is None:
        failures.append(f"Manifest lacks an artifact identity: {framework}")
    elif expected_lock_id.removeprefix("git:") not in lock_text:
        failures.append(f"Lockfile does not contain the manifest artifact identity: {framework}")
    return failures


def _probe(interpreter: Path, target: FrameworkTarget) -> dict[str, Any]:
    code = """
import importlib.metadata
import json
from pathlib import Path
import ml4t.backtest
import sys

distribution = importlib.metadata.distribution(sys.argv[1])
print(json.dumps({
    "version": distribution.version,
    "direct_url": distribution.read_text("direct_url.json"),
    "ml4t_path": str(Path(ml4t.backtest.__file__).resolve()),
    "python": sys.version.split()[0],
}, sort_keys=True))
"""
    result = subprocess.run(
        [str(interpreter), "-c", code, target.package],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError("Environment probe did not return an object")
    return payload


def verify_environment(
    framework: str,
    target: FrameworkTarget,
    *,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Verify installed package, source checkout, and private or engine identity."""
    failures = definition_failures(framework, target)
    if failures:
        raise ValueError("; ".join(failures))
    interpreter = environment_path(target, root) / "bin" / "python"
    if not interpreter.is_file():
        raise ValueError(f"Environment interpreter is missing: {interpreter}")
    payload = _probe(interpreter, target)
    if payload.get("version") != target.version and framework != "lean":
        raise ValueError(
            f"Installed {target.package} version differs: {payload.get('version')} != {target.version}"
        )
    ml4t_path = Path(str(payload.get("ml4t_path")))
    if not ml4t_path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Environment does not import the current checkout: {ml4t_path}")

    evidence: dict[str, Any] = {
        "framework": framework,
        "environment": str(environment_path(target, root)),
        "python": payload.get("python"),
        "package": target.package,
        "version": payload.get("version"),
        "immutable_id": target.immutable_id,
    }
    if framework == "vectorbt_pro":
        direct_url = json.loads(payload.get("direct_url") or "{}")
        commit = direct_url.get("vcs_info", {}).get("commit_id")
        if commit != target.source_commit:
            raise ValueError(f"Installed VectorBT Pro commit differs: {commit}")
        evidence["source_commit"] = commit
    if framework == "lean":
        if payload.get("version") != target.cli_version:
            raise ValueError(
                f"Installed LEAN CLI differs: {payload.get('version')} != {target.cli_version}"
            )
        if target.artifact is None:
            raise ValueError("LEAN target lacks an immutable engine image")
        subprocess.run(
            ["docker", "buildx", "imagetools", "inspect", target.artifact],
            check=True,
            capture_output=True,
            text=True,
        )
        evidence = {
            "framework": framework,
            "data_preparation": {
                "cli_package": target.package,
                "cli_version": payload.get("version"),
                "cli_immutable_id": target.cli_immutable_id,
                "python": payload.get("python"),
            },
            "engine_execution": {
                "image": target.artifact,
                "image_digest": target.immutable_id,
                "platform_artifact": target.platform_artifact,
            },
        }
    return evidence


def build_environment(
    framework: str,
    target: FrameworkTarget,
    *,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Synchronize one locked environment and verify its installed identity."""
    failures = definition_failures(framework, target)
    if failures:
        raise ValueError("; ".join(failures))
    destination = environment_path(target, root)
    command_environment = os.environ.copy()
    command_environment["UV_PROJECT_ENVIRONMENT"] = str(destination)
    command_environment["UV_NO_SOURCES"] = "0"
    try:
        subprocess.run(
            ["uv", "sync", "--project", str(project_directory(framework)), "--locked", "--no-dev"],
            cwd=PROJECT_ROOT,
            check=True,
            env=command_environment,
        )
    except subprocess.CalledProcessError as error:
        if framework == "vectorbt_pro":
            raise RuntimeError(
                "VectorBT Pro is unavailable. Authorized GitHub CLI access to the licensed "
                "source is required; no credentials are stored in this repository."
            ) from error
        raise
    return verify_environment(framework, target, root=root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--framework", choices=BUILDABLE_FRAMEWORKS)
    selection.add_argument("--all-public", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--environment-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--evidence-output", type=Path)
    args = parser.parse_args()

    manifest = load_framework_manifest()
    frameworks = PUBLIC_FRAMEWORKS if args.all_public else (args.framework,)
    evidence: list[dict[str, Any]] = []
    try:
        for framework in frameworks:
            if framework is None:
                continue
            target = manifest.targets[framework]
            result = (
                verify_environment(framework, target, root=args.environment_root)
                if args.verify_only
                else build_environment(framework, target, root=args.environment_root)
            )
            evidence.append(result)
            print(f"Verified {framework}: {json.dumps(result, sort_keys=True)}")
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Environment validation failed: {error}", file=sys.stderr)
        return 2

    if args.evidence_output is not None:
        args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
        args.evidence_output.write_text(
            json.dumps({"schema_version": 1, "environments": evidence}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
