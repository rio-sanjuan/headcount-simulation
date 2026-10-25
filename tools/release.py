"""Releases: the tag, the package's version, and the changelog agree before anything ships.

Versions follow Semantic Versioning, `MAJOR.MINOR.PATCH`, and a release's tag is the version
with a `v` before it. A release is made in two steps:

1. **Prepare** (`prepare X.Y.Z [--date YYYY-MM-DD]`): set the package's version and date its
   changelog section, `## [X.Y.Z] - YYYY-MM-DD`, in a PR of its own. A section that doesn't
   exist yet takes the changelog's `## [Unreleased]` entries.
2. **Publish**, once that's merged: a maintainer publishes a GitHub release tagged `vX.Y.Z`.
   The release workflows check it (`check`), build, scan, and attach the package, set the
   release's notes from the changelog (`notes`), and attach the small world.

`check vX.Y.Z` reports every disagreement at once: a tag that isn't a version, a package
version that differs, and a changelog section that's missing or undated. `notes vX.Y.Z` prints
the section's text.

Exit codes: 0 they agree, 1 they don't, 2 a file couldn't be read.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
CHANGELOG = ROOT / "CHANGELOG.md"
SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
#: The package's version, in pyproject.toml's `[project]` table.
PROJECT_VERSION = re.compile(r'(?m)^(\[project\][^\[]*?^version = ")([^"]+)(")', re.S)
UNRELEASED = "## [Unreleased]"


class ReleaseError(ValueError):
    """Files that can't be prepared or checked, every problem listed."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


def version_of(tag: str) -> str:
    """The version a `vX.Y.Z` tag names."""
    version = tag.removeprefix("v")
    if not tag.startswith("v") or not SEMVER.match(version):
        raise ReleaseError([f"a release's tag is v and a version, MAJOR.MINOR.PATCH, not {tag!r}"])
    return version


def package_version(pyproject: str) -> str:
    found = PROJECT_VERSION.search(pyproject)
    if found is None:
        raise ReleaseError(["pyproject.toml's [project] table has no version"])
    return found.group(2)


def _heading(version: str) -> re.Pattern[str]:
    return re.compile(rf"(?m)^## \[{re.escape(version)}\]( - (.*))?$")


def section(changelog: str, version: str) -> tuple[str | None, str]:
    """The date the changelog's section for `version` gives, if any, and its text."""
    found = _heading(version).search(changelog)
    if found is None:
        raise ReleaseError([f"the changelog has no section for {version}"])
    rest = changelog[found.end() :]
    following = re.search(r"(?m)^## \[", rest)
    text = rest[: following.start()] if following else rest
    return found.group(2), text.strip()


def problems(tag: str, pyproject: str, changelog: str) -> list[str]:
    """Every way the tag, the package's version, and the changelog disagree."""
    try:
        version = version_of(tag)
    except ReleaseError as error:
        return error.problems
    found = []
    try:
        package = package_version(pyproject)
        if package != version:
            found.append(f"the package's version is {package}, not the tag's {version}")
    except ReleaseError as error:
        found += error.problems
    try:
        date, text = section(changelog, version)
        if date is None or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            found.append(f"the changelog's {version} section is undated: prepare the release first")
        if not text:
            found.append(f"the changelog's {version} section says nothing")
    except ReleaseError as error:
        found += error.problems
    return found


def prepare(version: str, date: dt.date, pyproject: str, changelog: str) -> tuple[str, str]:
    """`pyproject` and `changelog` with the package at `version`, and the changelog's section
    for it dated `date`: the section it has, or a new one with the Unreleased entries."""
    if not SEMVER.match(version):
        raise ReleaseError([f"a version is MAJOR.MINOR.PATCH, not {version!r}"])
    if PROJECT_VERSION.search(pyproject) is None:
        raise ReleaseError(["pyproject.toml's [project] table has no version"])
    pyproject = PROJECT_VERSION.sub(rf"\g<1>{version}\g<3>", pyproject, count=1)
    dated = f"## [{version}] - {date.isoformat()}"
    heading = _heading(version)
    if heading.search(changelog):
        changelog = heading.sub(dated, changelog, count=1)
    elif UNRELEASED in changelog:
        changelog = changelog.replace(UNRELEASED, f"{UNRELEASED}\n\n{dated}", 1)
    else:
        raise ReleaseError([f"the changelog has no section for {version}, and no {UNRELEASED}"])
    return pyproject, changelog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "notes"):
        commands.add_parser(name).add_argument("tag")
    ready = commands.add_parser("prepare")
    ready.add_argument("version")
    ready.add_argument("--date", type=dt.date.fromisoformat, default=dt.date.today())
    args = parser.parse_args(argv)
    try:
        pyproject, changelog = PYPROJECT.read_text(), CHANGELOG.read_text()
    except OSError as error:
        print(f"release: {error}")
        return 2
    try:
        if args.command == "check":
            found = problems(args.tag, pyproject, changelog)
            for problem in found:
                print(f"release: {problem}")
            if not found:
                print(f"release: {args.tag} agrees with the package and the changelog")
            return 1 if found else 0
        if args.command == "notes":
            print(section(changelog, version_of(args.tag))[1])
            return 0
        pyproject, changelog = prepare(args.version, args.date, pyproject, changelog)
    except ReleaseError as error:
        for problem in error.problems:
            print(f"release: {problem}")
        return 1
    PYPROJECT.write_text(pyproject)
    CHANGELOG.write_text(changelog)
    print(f"release: prepared {args.version}, dated {args.date}; run `uv lock` and open a PR")
    return 0


if __name__ == "__main__":
    sys.exit(main())
