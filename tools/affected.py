"""Select the tests a change can affect, from the import graph.

The layering is strict, so the import graph shows exactly which layers a change can reach: a
change in a layer affects its own tests and the tests of every layer that imports it, directly
or not. A change in `hcsim.financial`, for example, runs the financial, engine, and studies
tests, and a change in `hcsim.model` runs everything. Any source change also runs the
end-to-end tests, which pass through every layer.

Prints a space-separated list of test paths for `make test TESTS=...`. Prints `tests` when
everything must run, and nothing when the change can't affect any test, such as docs only.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import grimp

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "hcsim"
ALL = ["tests"]

# Changes that can't affect any test.
NO_TESTS_PREFIXES = ("docs/",)
NO_TESTS_FILES = {"README.md", "CONTRIBUTING.md", "SECURITY.md", "NOTICE", "LICENSE"}
# Changes whose effect isn't confined to one layer, so everything runs.
ALL_TESTS_PREFIXES = (".github/", "tests/support/")
ALL_TESTS_FILES = {
    "pyproject.toml",
    "uv.lock",
    "Makefile",
    ".pre-commit-config.yaml",
    ".python-version",
    "tests/conftest.py",
    "tools/affected.py",
    f"src/{PACKAGE}/__init__.py",
}


def layer_of_source(path: str) -> str | None:
    """`src/hcsim/model/core/x.py` belongs to the `model` layer."""
    parts = Path(path).parts
    if len(parts) >= 4 and parts[0] == "src" and parts[1] == PACKAGE:
        return parts[2]
    return None


def select_tests(
    changed: Iterable[str],
    importers_of: Callable[[str], set[str]],
    existing: set[str],
) -> list[str]:
    """Pure selection logic: changed paths, a layer graph, and the test paths that exist."""
    selected: set[str] = set()
    layers: set[str] = set()
    for path in changed:
        if path in NO_TESTS_FILES or path.startswith(NO_TESTS_PREFIXES):
            continue
        if path in ALL_TESTS_FILES or path.startswith(ALL_TESTS_PREFIXES):
            return ALL
        if layer := layer_of_source(path):
            layers.add(layer)
        elif path.startswith("tests/"):
            parts = Path(path).parts
            selected.add(f"tests/{parts[1]}" if len(parts) > 2 else path)
        elif path.startswith("tools/"):
            selected.add("tests/tools")
        else:
            return ALL  # Unknown territory: run everything rather than guess.
    for layer in set(layers):
        layers |= importers_of(layer)
    if layers:
        selected |= {f"tests/{layer}" for layer in layers}
        selected |= {p for p in existing if p.startswith("tests/test_")}  # top-level smoke tests
        selected.add("tests/tools")  # the tools inspect the real package, such as its graph
        selected.add("tests/e2e")  # the end-to-end run passes through every layer
    return sorted(p for p in selected if p in existing)


def importers_from_graph(layer: str) -> set[str]:
    """Layers that import `layer`, directly or transitively, per the real import graph."""
    graph = grimp.build_graph(PACKAGE)
    downstream = graph.find_downstream_modules(f"{PACKAGE}.{layer}", as_package=True)
    return {module.split(".")[1] for module in downstream if module.count(".") >= 1} - {layer}


def existing_test_paths() -> set[str]:
    tests = ROOT / "tests"
    paths = {f"tests/{p.name}" for p in tests.iterdir() if p.is_dir() or p.suffix == ".py"}
    return paths | {"tests"}


def changed_paths(base: str, head: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="", help="base ref; empty means run everything")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args(argv)
    if not args.base:
        print(" ".join(ALL))
        return 0
    selected = select_tests(
        changed_paths(args.base, args.head), importers_from_graph, existing_test_paths()
    )
    print(" ".join(selected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
