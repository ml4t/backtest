"""Reject a release request unless its version and commit are unpublished main state."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import urllib.error
import urllib.request

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_STABLE_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


def preflight_failures(
    *,
    version: str,
    candidate_commit: str,
    workflow_commit: str,
    checkout_commit: str,
    main_commit: str,
    tag_exists: bool,
    release_exists: bool,
    pypi_exists: bool,
) -> list[str]:
    """Return every reason a requested release must not start qualification."""
    failures: list[str] = []
    if _STABLE_VERSION.fullmatch(version) is None:
        failures.append(f"Release version must be a stable X.Y.Z value: {version!r}")
    if _COMMIT.fullmatch(candidate_commit) is None:
        failures.append("Candidate commit must be a full lowercase Git SHA")
    for label, actual in (
        ("workflow", workflow_commit),
        ("checkout", checkout_commit),
        ("origin/main", main_commit),
    ):
        if actual != candidate_commit:
            failures.append(
                f"Candidate commit differs from {label}: {candidate_commit!r} != {actual!r}"
            )
    tag = f"v{version}"
    if tag_exists:
        failures.append(f"Git tag already exists: {tag}")
    if release_exists:
        failures.append(f"GitHub release already exists: {tag}")
    if pypi_exists:
        failures.append(f"PyPI version already exists: {version}")
    return failures


def _git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _remote_tag_exists(tag: str) -> bool:
    result = subprocess.run(
        ["git", "ls-remote", "--exit-code", "--tags", "origin", f"refs/tags/{tag}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 2:
        return False
    raise OSError(result.stderr.strip() or "Could not inspect remote Git tags")


def _url_exists(url: str, *, token: str | None = None) -> bool:
    headers = {"User-Agent": "ml4t-release-preflight"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            return response.status == 200
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--workflow-commit", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--pypi-name", required=True)
    parser.add_argument("--index-url", default="https://pypi.org/pypi")
    parser.add_argument("--github-api-url", default="https://api.github.com")
    args = parser.parse_args()

    if _REPOSITORY.fullmatch(args.repository) is None:
        parser.error("--repository must have owner/name form")

    try:
        checkout_commit = _git_output("rev-parse", "HEAD")
        main_commit = _git_output("rev-parse", "origin/main")
        tag = f"v{args.version}"
        failures = preflight_failures(
            version=args.version,
            candidate_commit=args.candidate_commit,
            workflow_commit=args.workflow_commit,
            checkout_commit=checkout_commit,
            main_commit=main_commit,
            tag_exists=_remote_tag_exists(tag),
            release_exists=_url_exists(
                f"{args.github_api_url.rstrip('/')}/repos/{args.repository}/releases/tags/{tag}",
                token=os.environ.get("GITHUB_TOKEN"),
            ),
            pypi_exists=_url_exists(
                f"{args.index_url.rstrip('/')}/{args.pypi_name}/{args.version}/json"
            ),
        )
    except (OSError, subprocess.SubprocessError, urllib.error.URLError) as error:
        print(f"Release preflight could not establish external state: {error}")
        return 1

    for failure in failures:
        print(failure)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
