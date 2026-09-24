"""Check links in rendered documentation content, including section anchors."""

from __future__ import annotations

import argparse
import posixpath
import time
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import Request, urlopen

_SITE_ORIGIN = "https://www.ml4trading.io"
_SITE_PREFIX = "/docs/backtest/"
_USER_AGENT = "ml4t-backtest-documentation-link-check/1.0"
_GUIDE_SECTIONS = {"user-guide", "book-guide"}


class ContentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_article = False
        self.links: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "article":
            self.in_article = True
        if identifier := values.get("id"):
            self.ids.add(identifier)
        if self.in_article:
            attribute = "src" if tag == "img" else "href" if tag == "a" else None
            if attribute and (link := values.get(attribute)):
                self.links.append(link)

    def handle_endtag(self, tag: str) -> None:
        if tag == "article":
            self.in_article = False


def _page_url(path: Path, site: Path) -> str:
    relative = path.relative_to(site).as_posix()
    if relative == "index.html":
        return _SITE_ORIGIN + _SITE_PREFIX
    if relative.endswith("/index.html"):
        return _SITE_ORIGIN + _SITE_PREFIX + relative[: -len("index.html")]
    return _SITE_ORIGIN + _SITE_PREFIX + relative


def _target_path(site: Path, path: str) -> Path:
    relative = unquote(path.removeprefix(_SITE_PREFIX))
    normalized = posixpath.normpath(relative).lstrip("/")
    if path.endswith("/") or normalized == ".":
        normalized = posixpath.join(normalized, "index.html")
    target = (site / normalized).resolve()
    if not target.is_relative_to(site.resolve()):
        raise ValueError("link escapes the documentation site")
    return target


def _check_external(url: str) -> str | None:
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": _USER_AGENT})
            with urlopen(request, timeout=15) as response:
                if response.status >= 400:
                    return f"HTTP {response.status}"
            return None
        except (HTTPError, URLError, TimeoutError) as error:
            if attempt == 2 or (isinstance(error, HTTPError) and error.code < 500):
                return str(error)
            time.sleep(attempt + 1)
    return "unreachable"


def check_links(site: Path) -> tuple[int, int]:
    site = site.resolve()
    pages: dict[Path, ContentParser] = {}
    for page in sorted(site.rglob("*.html")):
        parser = ContentParser()
        parser.feed(page.read_text(encoding="utf-8"))
        pages[page.resolve()] = parser

    failures: list[str] = []
    external: set[str] = set()
    checked = 0
    for page, content in pages.items():
        if page.relative_to(site).parts[0] not in _GUIDE_SECTIONS:
            continue
        for link in content.links:
            checked += 1
            destination = urlsplit(urljoin(_page_url(page, site), link))
            if destination.scheme not in {"http", "https"}:
                failures.append(f"{page.relative_to(site)}: unsupported link {link}")
                continue
            if destination.netloc not in {"www.ml4trading.io", "ml4trading.io"}:
                external.add(destination._replace(fragment="").geturl())
                continue
            if not destination.path.startswith(_SITE_PREFIX):
                external.add(destination._replace(fragment="").geturl())
                continue
            try:
                target = _target_path(site, destination.path)
            except ValueError as error:
                failures.append(f"{page.relative_to(site)}: {link}: {error}")
                continue
            if not target.is_file():
                failures.append(f"{page.relative_to(site)}: {link}: missing target")
                continue
            if destination.fragment and target.suffix == ".html":
                anchor = unquote(destination.fragment)
                if anchor not in pages[target].ids:
                    failures.append(f"{page.relative_to(site)}: {link}: missing anchor")

    with ThreadPoolExecutor(max_workers=8) as executor:
        for url, error in zip(sorted(external), executor.map(_check_external, sorted(external))):
            if error:
                failures.append(f"{url}: {error}")
    if failures:
        raise ValueError("Documentation link failures:\n" + "\n".join(failures))
    return checked, len(external)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, default=Path("site"))
    args = parser.parse_args()
    checked, external = check_links(args.site)
    print(f"Checked {checked} rendered content links and {external} external destinations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
