"""Verify rendered and deployed documentation release identity."""

from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

REQUIRED_META = ("ml4t-library", "ml4t-version", "ml4t-commit")


class _MetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta":
            return
        values = dict(attrs)
        name = values.get("name")
        content = values.get("content")
        if name in REQUIRED_META and content is not None:
            self.values[name] = content


def identity_failures(
    html: str,
    *,
    expected_library: str,
    expected_version: str,
    expected_commit: str,
    source: str,
) -> list[str]:
    parser = _MetadataParser()
    parser.feed(html)
    expected = {
        "ml4t-library": expected_library,
        "ml4t-version": expected_version,
        "ml4t-commit": expected_commit,
    }
    return [
        f"{source}: {name} is {parser.values.get(name)!r}, expected {value!r}"
        for name, value in expected.items()
        if parser.values.get(name) != value
    ]


def _site_pages(site: Path) -> list[Path]:
    return sorted(
        path
        for path in site.rglob("*.html")
        if path.is_file() and "overrides" not in path.relative_to(site).parts
    )


def _read_url(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "ml4t-release-verifier"})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        if response.status != 200:
            raise ValueError(f"{url}: HTTP {response.status}")
        return response.read().decode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--site", type=Path)
    source.add_argument("--url", action="append")
    parser.add_argument("--expected-library", required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()

    if len(args.expected_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.expected_commit
    ):
        parser.error("--expected-commit must be a full lowercase Git SHA")

    failures: list[str] = []
    try:
        if args.site is not None:
            pages = _site_pages(args.site)
            if not pages:
                failures.append(f"{args.site}: no rendered HTML pages found")
            for page in pages:
                failures.extend(
                    identity_failures(
                        page.read_text(encoding="utf-8"),
                        expected_library=args.expected_library,
                        expected_version=args.expected_version,
                        expected_commit=args.expected_commit,
                        source=str(page),
                    )
                )
        else:
            for url in args.url:
                failures.extend(
                    identity_failures(
                        _read_url(url),
                        expected_library=args.expected_library,
                        expected_version=args.expected_version,
                        expected_commit=args.expected_commit,
                        source=url,
                    )
                )
    except (OSError, UnicodeDecodeError, urllib.error.URLError, ValueError) as error:
        failures.append(str(error))

    for failure in failures:
        print(failure)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
