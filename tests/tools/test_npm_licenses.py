"""Tests for the npm license check."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.npm_licenses import check, main, verdict

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("license_", "expected"),
    [
        ("MIT", "ok"),
        ("MPL-2.0", "ok"),
        ("(MIT OR GPL-3.0)", "ok"),
        ("GPL-3.0-only", "incompatible"),
        ("AGPL-3.0 OR SSPL-1.0", "incompatible"),
        ("", "review"),
    ],
)
def test_a_license_is_judged_as_the_python_check_judges_it(license_: str, expected: str) -> None:
    assert verdict(license_) == expected


def test_every_package_in_a_lockfile_is_judged() -> None:
    lock = {
        "packages": {
            "": {"name": "root"},
            "node_modules/a": {"version": "1.0.0", "license": "MIT"},
            "node_modules/a/node_modules/b": {"version": "2.0.0", "license": "GPL-2.0"},
            "node_modules/c": {"version": "3.0.0"},
        }
    }
    assert check(lock) == {
        "a@1.0.0": ("MIT", "ok"),
        "b@2.0.0": ("GPL-2.0", "incompatible"),
        "c@3.0.0": ("", "review"),
    }
    with pytest.raises(ValueError, match="lists its packages"):
        check({"lockfileVersion": 1})


def test_the_command_fails_on_copyleft_and_reads_the_explorers_lockfile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(ROOT / "apps" / "explorer" / "package-lock.json")]) == 0
    assert "0 incompatible" in capsys.readouterr().out
    bad = tmp_path / "lock.json"
    bad.write_text(
        json.dumps({"packages": {"node_modules/x": {"version": "1", "license": "AGPL-3.0"}}})
    )
    assert main([str(bad)]) == 1
    assert "incompatible x@1: AGPL-3.0" in capsys.readouterr().out.replace("  ", " ").replace(
        "  ", " "
    )
    assert main([str(tmp_path / "missing.json")]) == 2
