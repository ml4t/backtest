"""Rendered guide links must resolve to pages, sections, and reachable URLs."""

from __future__ import annotations

import importlib.util
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

_ROOT = Path(__file__).parents[2]


def _load_checker():
    path = _ROOT / "validation" / "check_documentation_links.py"
    spec = importlib.util.spec_from_file_location("ml4t_documentation_links", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_site(site: Path, href: str) -> None:
    page = site / "user-guide" / "index.html"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(f'<article><a href="{href}">Go</a></article>')
    destination = site / "tutorials" / "index.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('<article><h2 id="working-section">Working</h2></article>')


def test_rendered_page_and_fragment_links(tmp_path: Path) -> None:
    checker = _load_checker()
    _write_site(tmp_path, "../tutorials/#working-section")
    assert checker.check_links(tmp_path) == (1, 0)

    _write_site(tmp_path, "../tutorials/#missing-section")
    with pytest.raises(ValueError, match="missing anchor"):
        checker.check_links(tmp_path)

    _write_site(tmp_path, "../absent/")
    with pytest.raises(ValueError, match="missing target"):
        checker.check_links(tmp_path)


def test_external_url_rejects_http_404(tmp_path: Path) -> None:
    checker = _load_checker()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200 if self.path == "/present" else 404)
            self.end_headers()

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        _write_site(tmp_path, base + "/present")
        assert checker.check_links(tmp_path) == (1, 1)
        _write_site(tmp_path, base + "/absent")
        with pytest.raises(ValueError, match="HTTP Error 404"):
            checker.check_links(tmp_path)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
