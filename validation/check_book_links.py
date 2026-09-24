"""Verify pinned companion-repository links against a checked Git tree."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

_ROOT = Path(__file__).parents[1]
_MANIFEST = _ROOT / "validation" / "book_link_manifest.json"
_COMPANION_URL = "https://github.com/stefan-jansen/machine-learning-for-trading"
_BLOB_URL = re.compile(
    re.escape(_COMPANION_URL) + r"/blob/(?P<commit>[0-9a-f]{40})/(?P<path>[^)\s#?]+)"
)
_MARKDOWN_URL = re.compile(
    r"\]\((https://github\.com/stefan-jansen/machine-learning-for-trading[^)]+)\)"
)


def documentation_paths() -> list[Path]:
    return sorted((_ROOT / "docs").rglob("*.md")) + [_ROOT / "README.md"]


def companion_links(paths: list[Path]) -> dict[str, str]:
    """Return linked companion paths and reject mutable or unsupported URL forms."""
    links: dict[str, str] = {}
    for page in paths:
        for match in _MARKDOWN_URL.finditer(page.read_text(encoding="utf-8")):
            url = match.group(1)
            if url == _COMPANION_URL or url == _COMPANION_URL + "/":
                continue
            blob = _BLOB_URL.fullmatch(url)
            if blob is None:
                raise ValueError(f"Unpinned companion URL in {page}: {url}")
            path = unquote(blob.group("path"))
            previous = links.setdefault(path, blob.group("commit"))
            if previous != blob.group("commit"):
                raise ValueError(f"Multiple companion revisions for {path}")
    return links


def check_links(paths: list[Path], manifest_path: Path = _MANIFEST) -> None:
    """Check every linked path against the manifest generated from the companion Git tree."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    commit = manifest["commit"]
    blobs = manifest["blobs"]
    links = companion_links(paths)
    for path, revision in links.items():
        if revision != commit:
            raise ValueError(f"Companion link uses {revision}, expected {commit}: {path}")
        if path not in blobs:
            raise ValueError(f"Companion path does not exist at {commit}: {path}")
    if paths == documentation_paths() and set(links) != set(blobs):
        raise ValueError("Book link manifest does not match the current documentation links")
    print(f"Checked {len(links)} companion paths at {commit}")


def write_manifest(companion_repo: Path) -> None:
    """Record linked blob identities from the checked companion revision."""
    commit = subprocess.check_output(
        ["git", "-C", str(companion_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    links = companion_links(documentation_paths())
    if set(links.values()) != {commit}:
        raise ValueError(f"All companion links must use checked revision {commit}")
    tree = subprocess.check_output(
        ["git", "-C", str(companion_repo), "ls-tree", "-r", commit], text=True
    )
    available: dict[str, str] = {}
    for line in tree.splitlines():
        metadata, path = line.split("\t", 1)
        mode, object_type, object_id = metadata.split()
        if mode in {"100644", "100755"} and object_type == "blob":
            available[path] = object_id
    missing = set(links) - set(available)
    if missing:
        raise ValueError(f"Companion paths absent from {commit}: {sorted(missing)}")
    manifest = {"commit": commit, "blobs": {path: available[path] for path in sorted(links)}}
    _MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {len(links)} companion paths at {commit}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--companion-repo", type=Path)
    args = parser.parse_args()
    if args.write:
        if args.paths or args.companion_repo is None:
            parser.error("--write requires --companion-repo and no path arguments")
        write_manifest(args.companion_repo)
    else:
        check_links(args.paths or documentation_paths())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
