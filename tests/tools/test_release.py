"""Tests for the release checks."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from tools import release
from tools.release import ReleaseError, package_version, prepare, problems, section, version_of

PYPROJECT = '[project]\nname = "hcsim"\nversion = "0.1.0.dev0"\n\n[tool.x]\nversion = "9"\n'
CHANGELOG = """# Changelog

## [Unreleased]

- a fix to come

## [0.1.0] - Unreleased

### Added

- everything
"""
DAY = dt.date(2026, 11, 2)


def test_a_tag_is_v_and_a_semantic_version() -> None:
    assert version_of("v1.2.3") == "1.2.3"
    for bad in ("1.2.3", "v1.2", "v01.2.3", "v1.2.3-rc.1"):
        with pytest.raises(ReleaseError, match=r"MAJOR\.MINOR\.PATCH"):
            version_of(bad)


def test_the_package_version_is_the_project_tables() -> None:
    assert package_version(PYPROJECT) == "0.1.0.dev0"
    with pytest.raises(ReleaseError, match="no version"):
        package_version("[tool.x]\nversion = '1'\n")


def test_an_unprepared_release_is_refused_with_every_problem() -> None:
    assert problems("v0.1.0", PYPROJECT, CHANGELOG) == [
        "the package's version is 0.1.0.dev0, not the tag's 0.1.0",
        "the changelog's 0.1.0 section is undated: prepare the release first",
    ]
    assert problems("v0.2.0", PYPROJECT, CHANGELOG)[-1] == "the changelog has no section for 0.2.0"
    assert problems("0.1.0", PYPROJECT, CHANGELOG) == [
        "a release's tag is v and a version, MAJOR.MINOR.PATCH, not '0.1.0'"
    ]


def test_preparing_sets_the_version_and_dates_the_section() -> None:
    pyproject, changelog = prepare("0.1.0", DAY, PYPROJECT, CHANGELOG)
    assert package_version(pyproject) == "0.1.0"
    assert 'version = "9"' in pyproject  # another table's version is left alone
    assert "## [0.1.0] - 2026-11-02" in changelog
    assert problems("v0.1.0", pyproject, changelog) == []
    date, text = section(changelog, "0.1.0")
    assert date == "2026-11-02"
    assert text == "### Added\n\n- everything"


def test_a_new_version_takes_the_unreleased_entries() -> None:
    pyproject, changelog = prepare("0.1.1", DAY, PYPROJECT, CHANGELOG)
    assert package_version(pyproject) == "0.1.1"
    assert changelog.index("## [Unreleased]") < changelog.index("## [0.1.1] - 2026-11-02")
    assert section(changelog, "0.1.1")[1] == "- a fix to come"
    with pytest.raises(ReleaseError, match=r"MAJOR\.MINOR\.PATCH"):
        prepare("0.1", DAY, PYPROJECT, CHANGELOG)
    with pytest.raises(ReleaseError, match=r"no section for 0\.2\.0, and no"):
        prepare("0.2.0", DAY, PYPROJECT, "# Changelog\n")
    with pytest.raises(ReleaseError, match="no version"):
        prepare("0.2.0", DAY, "[tool.x]\n", CHANGELOG)


def test_an_empty_section_says_nothing() -> None:
    dated = "## [0.1.0] - 2026-11-02\n\n## [0.0.1] - 2026-01-01\n\n- old\n"
    pyproject = PYPROJECT.replace("0.1.0.dev0", "0.1.0")
    assert problems("v0.1.0", pyproject, dated) == ["the changelog's 0.1.0 section says nothing"]


def test_the_command_checks_prints_and_prepares(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pyproject, changelog = tmp_path / "pyproject.toml", tmp_path / "CHANGELOG.md"
    pyproject.write_text(PYPROJECT)
    changelog.write_text(CHANGELOG)
    monkeypatch.setattr(release, "PYPROJECT", pyproject)
    monkeypatch.setattr(release, "CHANGELOG", changelog)
    assert release.main(["check", "v0.1.0"]) == 1
    assert "undated" in capsys.readouterr().out
    assert release.main(["prepare", "0.1.0", "--date", "2026-11-02"]) == 0
    assert release.main(["check", "v0.1.0"]) == 0
    assert "agrees" in capsys.readouterr().out
    assert release.main(["notes", "v0.1.0"]) == 0
    assert capsys.readouterr().out.strip() == "### Added\n\n- everything"
    assert release.main(["prepare", "1.0"]) == 1
    assert release.main(["notes", "v9.0.0"]) == 1
    monkeypatch.setattr(release, "CHANGELOG", tmp_path / "missing.md")
    assert release.main(["check", "v0.1.0"]) == 2


def test_the_repositorys_changelog_holds_the_next_release() -> None:
    root = Path(__file__).resolve().parents[2]
    changelog = (root / "CHANGELOG.md").read_text()
    pyproject = (root / "pyproject.toml").read_text()
    version = package_version(pyproject).removesuffix(".dev0")
    _, text = section(changelog, version)
    assert text
    assert "## [Unreleased]" in changelog
