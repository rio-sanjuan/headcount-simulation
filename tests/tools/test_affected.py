"""Tests for test selection from the import graph."""

from __future__ import annotations

import pytest

from tools import affected
from tools.affected import ALL, importers_from_graph, select_tests

# A toy layer graph with the real shape: model is the leaf, financial imports model, and so on.
IMPORTERS = {
    "model": {"guidance", "financial", "engine", "studies"},
    "financial": {"engine", "studies"},
    "guidance": {"studies"},
    "engine": {"studies"},
    "studies": set(),
    "synth": set(),
}
EXISTING = {
    "tests",
    "tests/model",
    "tests/financial",
    "tests/engine",
    "tests/studies",
    "tests/tools",
    "tests/support",
    "tests/test_package.py",
}


def select(*changed: str) -> list[str]:
    return select_tests(changed, IMPORTERS.__getitem__, EXISTING)


def test_a_financial_change_runs_financial_and_its_importers() -> None:
    assert select("src/hcsim/financial/expense.py") == [
        "tests/engine",
        "tests/financial",
        "tests/studies",
        "tests/test_package.py",
    ]


def test_a_model_change_runs_every_layer_with_tests() -> None:
    assert select("src/hcsim/model/core/schema.py") == [
        "tests/engine",
        "tests/financial",
        "tests/model",
        "tests/studies",
        "tests/test_package.py",
    ]


def test_a_layer_without_a_test_directory_still_runs_the_smoke_tests() -> None:
    assert select("src/hcsim/synth/world.py") == ["tests/test_package.py"]


def test_docs_only_changes_run_nothing() -> None:
    assert select("docs/design/overview.md", "README.md") == []


@pytest.mark.parametrize(
    "path",
    [
        "pyproject.toml",
        "uv.lock",
        ".github/workflows/ci.yml",
        "tests/conftest.py",
        "tests/support/stat_check.py",
        "src/hcsim/__init__.py",
        "some/unknown/file.txt",
    ],
)
def test_cross_cutting_changes_run_everything(path: str) -> None:
    assert select(path, "src/hcsim/financial/x.py") == ALL


def test_test_and_tool_changes_run_their_own_tests() -> None:
    assert select("tests/engine/test_loop.py", "tools/sanitize_scan.py") == [
        "tests/engine",
        "tests/tools",
    ]


def test_the_real_graph_has_no_importers_of_an_empty_layer() -> None:
    # Every layer is still empty in M0, so no layer imports another yet.
    assert importers_from_graph("model") == set()


def test_main_without_a_base_runs_everything(capsys: pytest.CaptureFixture[str]) -> None:
    assert affected.main([]) == 0
    assert capsys.readouterr().out.strip() == "tests"


def test_main_with_a_base_selects_from_the_diff(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(affected, "changed_paths", lambda base, head: ["tools/licenses.py"])
    assert affected.main(["--base", "abc123"]) == 0
    assert capsys.readouterr().out.strip() == "tests/tools"


def test_the_real_repository_has_these_test_paths() -> None:
    assert {"tests", "tests/tools", "tests/support"} <= affected.existing_test_paths()


def test_a_diff_of_head_against_itself_is_empty() -> None:
    assert affected.changed_paths("HEAD", "HEAD") == []
