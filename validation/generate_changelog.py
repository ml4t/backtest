"""Generate managed release-history entries from structured release data."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parents[1]
_SOURCE = _ROOT / "release" / "changelog.json"
_OUTPUT = _ROOT / "CHANGELOG.md"
_HEADER = "# Changelog"
_MARKER = "<!-- Generated from release/changelog.json. Do not edit managed entries. -->"


def _load_source() -> dict[str, Any]:
    payload = json.loads(_SOURCE.read_text(encoding="utf-8"))
    releases = payload.get("releases")
    if not isinstance(releases, list) or not releases:
        raise ValueError("release/changelog.json requires a non-empty releases list")
    versions: set[str] = set()
    for release in releases:
        if not isinstance(release, dict):
            raise ValueError("Each changelog release must be an object")
        version = release.get("version")
        sections = release.get("sections")
        if not isinstance(version, str) or not version or version in versions:
            raise ValueError(f"Duplicate or invalid changelog version: {version!r}")
        versions.add(version)
        if not isinstance(sections, dict) or "Compatibility" not in sections:
            raise ValueError(f"Release {version} requires a Compatibility section")
    return payload


def _legacy_history(source: dict[str, Any]) -> str:
    legacy_start = source.get("legacy_start")
    if not isinstance(legacy_start, str) or not legacy_start:
        raise ValueError("release/changelog.json requires legacy_start")
    current = _OUTPUT.read_text(encoding="utf-8")
    heading = f"## {legacy_start} -"
    offset = current.find(heading)
    if offset < 0:
        raise ValueError(f"Could not find legacy changelog heading: {heading}")
    return current[offset:].strip()


def _bullet(item: str) -> str:
    lines = textwrap.wrap(
        item,
        width=98,
        initial_indent="- ",
        subsequent_indent="  ",
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\n".join(lines)


def render() -> str:
    source = _load_source()
    blocks = [_HEADER, "", _MARKER]
    for release in source["releases"]:
        blocks.extend(["", f"## {release['version']} - {release['date']}"])
        for section, items in release["sections"].items():
            if not isinstance(section, str) or not isinstance(items, list) or not items:
                raise ValueError(f"Invalid section in release {release['version']}: {section!r}")
            if not all(isinstance(item, str) and item for item in items):
                raise ValueError(f"Invalid items in release {release['version']}: {section}")
            blocks.extend(["", f"### {section}", "", *[_bullet(item) for item in items]])
    blocks.extend(["", _legacy_history(source), ""])
    return "\n".join(blocks)


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    generated = render()
    if arguments.check:
        if _OUTPUT.read_text(encoding="utf-8") != generated:
            print("CHANGELOG.md differs from release/changelog.json")
            return 1
        return 0
    _OUTPUT.write_text(generated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
