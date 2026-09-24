"""Execute selected Markdown examples with the active Python installation."""

from __future__ import annotations

import argparse
import difflib
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

_ROOT = Path(__file__).parents[1]
_DEFAULT_PATHS = (
    _ROOT / "README.md",
    _ROOT / "docs" / "index.md",
    _ROOT / "docs" / "getting-started" / "installation.md",
    _ROOT / "docs" / "getting-started" / "quickstart.md",
    _ROOT / "docs" / "tutorials" / "data.md",
    _ROOT / "docs" / "tutorials" / "orders-and-timing.md",
    _ROOT / "docs" / "tutorials" / "multiasset-rebalancing.md",
    _ROOT / "docs" / "tutorials" / "costs-and-funding.md",
    _ROOT / "docs" / "tutorials" / "accounts-and-constraints.md",
    _ROOT / "docs" / "tutorials" / "risk-and-state.md",
    _ROOT / "docs" / "tutorials" / "profiles-and-parity.md",
    _ROOT / "docs" / "tutorials" / "results-and-analysis.md",
    _ROOT / "docs" / "user-guide" / "accounts.md",
    _ROOT / "docs" / "user-guide" / "execution-semantics.md",
)
_REQUIRED_EXAMPLES = frozenset(
    {
        "account-engine-direct",
        "account-gatekeeper",
        "installation-import",
        "preopen-mixed-rules",
        "readme-quickstart",
        "home-example",
        "home-convenience",
        "quickstart-minimal",
        "smoke-equity",
        "smoke-etf",
        "smoke-future",
        "smoke-fx",
        "smoke-crypto-perp",
        "tutorial-orders-timing",
        "tutorial-multiasset-equity-etf",
        "tutorial-multiasset-future",
        "tutorial-multiasset-fx",
        "tutorial-costs-equity",
        "tutorial-costs-funding",
        "tutorial-costs-flip",
        "tutorial-accounts-constraints",
        "tutorial-risk-state",
        "tutorial-profiles-inspect",
        "tutorial-profiles-one-setting",
        "tutorial-results-export",
    }
)
_EXAMPLE = re.compile(
    r"<!-- ml4t-doc-test: (?P<name>[a-z0-9-]+) -->\s*"
    r"```(?P<language>python|bash)\n(?P<code>.*?)\n```",
    flags=re.DOTALL,
)
_OUTPUT = re.compile(
    r"<!-- ml4t-doc-output: (?P<name>[a-z0-9-]+) -->\s*"
    r"```text\n(?P<output>.*?)\n```",
    flags=re.DOTALL,
)


@dataclass(frozen=True)
class Example:
    name: str
    language: str
    code: str
    source: Path
    expected_output: str | None = None


def collect_examples(
    paths: list[Path] | tuple[Path, ...], *, require_all: bool = False
) -> list[Example]:
    """Collect uniquely named executable examples from Markdown files."""
    examples: list[Example] = []
    names: set[str] = set()
    for path in paths:
        content = path.read_text(encoding="utf-8")
        outputs: dict[str, str] = {}
        for match in _OUTPUT.finditer(content):
            name = match["name"]
            if name in outputs:
                raise ValueError(f"Duplicate documentation output name: {name}")
            outputs[name] = match["output"] + "\n"
        page_names: set[str] = set()
        for match in _EXAMPLE.finditer(content):
            name = match["name"]
            if name in names:
                raise ValueError(f"Duplicate documentation example name: {name}")
            names.add(name)
            page_names.add(name)
            if name.startswith(("smoke-", "tutorial-")) and name not in outputs:
                raise ValueError(f"Documentation example {name} has no recorded output")
            examples.append(
                Example(
                    name=name,
                    language=match["language"],
                    code=match["code"],
                    source=path,
                    expected_output=outputs.get(name),
                )
            )
        unknown = outputs.keys() - page_names
        if unknown:
            raise ValueError(f"Documentation outputs have no example in {path}: {sorted(unknown)}")
    if require_all:
        missing = sorted(_REQUIRED_EXAMPLES - names)
        if missing:
            raise ValueError(f"Required documentation examples are missing: {missing}")
    return examples


def run_examples(examples: list[Example]) -> None:
    """Run each example outside the source checkout import path."""
    with tempfile.TemporaryDirectory(prefix="ml4t-backtest-docs-") as directory:
        workdir = Path(directory)
        environment = os.environ.copy()
        environment["PATH"] = os.pathsep.join(
            [str(Path(sys.executable).parent), environment.get("PATH", "")]
        )
        environment.pop("PYTHONPATH", None)

        for example in examples:
            display_source = (
                example.source.relative_to(_ROOT)
                if example.source.is_relative_to(_ROOT)
                else example.source
            )
            print(f"Running {example.name} from {display_source}")
            if example.language == "python":
                script = workdir / f"{example.name}.py"
                script.write_text(example.code + "\n", encoding="utf-8")
                command = [sys.executable, "-I", str(script)]
            else:
                command = ["bash", "-euo", "pipefail", "-c", example.code]
            completed = subprocess.run(
                command,
                cwd=workdir,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if completed.returncode:
                raise RuntimeError(
                    f"Documentation example {example.name} failed:\n"
                    f"{completed.stdout}{completed.stderr}"
                )
            if example.expected_output is not None:
                expected = example.expected_output.replace(
                    "{package_version}", version("ml4t-backtest")
                )
                if completed.stdout != expected:
                    diff = "".join(
                        difflib.unified_diff(
                            expected.splitlines(keepends=True),
                            completed.stdout.splitlines(keepends=True),
                            fromfile="documented",
                            tofile="executed",
                        )
                    )
                    raise AssertionError(
                        f"Documentation example {example.name} output differs:\n{diff}"
                    )
            elif completed.stdout:
                print(completed.stdout, end="")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    arguments = parser.parse_args()
    paths = arguments.paths or list(_DEFAULT_PATHS)
    run_examples(collect_examples(paths, require_all=not arguments.paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
