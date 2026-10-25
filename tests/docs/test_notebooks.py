"""The doc notebooks run as tests: each one's code cells, in order, in a fresh namespace, so a
worked example that stops matching the engine fails here. They're committed without outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

NOTEBOOKS = sorted((Path(__file__).parents[2] / "docs" / "notebooks").glob("*.ipynb"))


def cells(path: Path) -> list[dict[str, Any]]:
    return list(json.loads(path.read_text())["cells"])


def test_there_are_notebooks() -> None:
    assert [p.name for p in NOTEBOOKS] == [
        "annual-grant.ipynb",
        "guided-study.ipynb",
        "quota-sampling.ipynb",
        "three-path-tranche.ipynb",
    ]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_a_notebook_is_committed_without_outputs(path: Path) -> None:
    for cell in cells(path):
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_a_notebook_runs(path: Path) -> None:
    namespace: dict[str, Any] = {"__name__": f"notebook_{path.stem.replace('-', '_')}"}
    for k, cell in enumerate(cells(path)):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"{path.name}[{k}]", "exec"), namespace)
