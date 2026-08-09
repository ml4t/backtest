"""Execute selected Markdown examples with the active Python installation."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).parents[1]
_DEFAULT_PATHS = (
    _ROOT / "README.md",
    _ROOT / "docs" / "getting-started" / "installation.md",
    _ROOT / "docs" / "getting-started" / "quickstart.md",
)
_REQUIRED_EXAMPLES = frozenset({"installation-import", "readme-quickstart", "quickstart-minimal"})
_EXAMPLE = re.compile(
    r"<!-- ml4t-doc-test: (?P<name>[a-z0-9-]+) -->\s*"
    r"```(?P<language>python|bash)\n(?P<code>.*?)\n```",
    flags=re.DOTALL,
)


@dataclass(frozen=True)
class Example:
    name: str
    language: str
    code: str
    source: Path


def collect_examples(
    paths: list[Path] | tuple[Path, ...], *, require_all: bool = False
) -> list[Example]:
    """Collect uniquely named executable examples from Markdown files."""
    examples: list[Example] = []
    names: set[str] = set()
    for path in paths:
        for match in _EXAMPLE.finditer(path.read_text(encoding="utf-8")):
            name = match["name"]
            if name in names:
                raise ValueError(f"Duplicate documentation example name: {name}")
            names.add(name)
            examples.append(
                Example(
                    name=name,
                    language=match["language"],
                    code=match["code"],
                    source=path,
                )
            )
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
            print(f"Running {example.name} from {example.source.relative_to(_ROOT)}")
            if example.language == "python":
                script = workdir / f"{example.name}.py"
                script.write_text(example.code + "\n", encoding="utf-8")
                command = [sys.executable, "-I", str(script)]
            else:
                command = ["bash", "-euo", "pipefail", "-c", example.code]
            subprocess.run(command, cwd=workdir, env=environment, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    arguments = parser.parse_args()
    paths = arguments.paths or list(_DEFAULT_PATHS)
    run_examples(collect_examples(paths, require_all=not arguments.paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
