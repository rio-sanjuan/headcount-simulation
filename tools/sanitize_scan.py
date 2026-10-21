"""Scan text for denylisted patterns without ever revealing what matched.

The denylist is private and never committed. This module is the one implementation of its
format: one extended regular expression per line, matched case-insensitively. A line starting
with ``#`` is a comment, and whitespace followed by ``#`` starts an inline comment.

Patterns are read from the first of these that exists:

1. ``--patterns PATH``
2. ``SANITIZE_DENYLIST``, which holds the list itself (CI reads it from a secret)
3. ``HCSIM_DENYLIST``, which holds a path
4. ``~/.config/hcsim/denylist.txt``

With none of them available, the scan fails rather than passing silently. Reports name a
location and a denylist line number, never the matched text, because CI logs on a public
repository are public.

Exit codes: 0 clean, 1 matches found, 2 usage or configuration error.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path("~/.config/hcsim/denylist.txt").expanduser()
# Private material that must never enter the repository, matched against file names.
FORBIDDEN_NAMES = (re.compile(r"(^|/)project-description\.md$"), re.compile(r"denylist[^/]*\.txt$"))
ZERO_SHA = re.compile(r"^0+$")
BINARY_SNIFF_BYTES = 8192


class DenylistError(Exception):
    """The denylist is missing or malformed."""


@dataclass(frozen=True)
class Pattern:
    line: int
    regex: re.Pattern[str]


@dataclass(frozen=True)
class Hit:
    location: str
    line: int | None
    pattern_line: int | None
    note: str = ""

    def __str__(self) -> str:
        where = self.location if self.line is None else f"{self.location}:{self.line}"
        what = f"denylist line {self.pattern_line}" if self.pattern_line else "forbidden name"
        return f"{where}: {what}{f' ({self.note})' if self.note else ''}"


def parse_patterns(text: str) -> list[Pattern]:
    patterns = []
    for number, raw in enumerate(text.splitlines(), 1):
        line = re.sub(r"\s+#.*$", "", raw).strip()
        if not line or line.startswith("#"):
            continue
        try:
            patterns.append(Pattern(number, re.compile(line, re.IGNORECASE)))
        except re.error:
            # `from None` keeps the re.error message, which quotes the pattern, out of tracebacks.
            raise DenylistError(
                f"denylist line {number} is not a valid regular expression"
            ) from None
    if not patterns:
        raise DenylistError("the denylist contains no patterns")
    return patterns


def load_patterns(path: Path | None = None, env: dict[str, str] | None = None) -> list[Pattern]:
    env = dict(os.environ) if env is None else env
    if path is not None:
        if not path.is_file():
            raise DenylistError(f"no denylist at {path}")
        return parse_patterns(path.read_text())
    if env.get("SANITIZE_DENYLIST", "").strip():
        return parse_patterns(env["SANITIZE_DENYLIST"])
    if env.get("HCSIM_DENYLIST"):
        candidate = Path(env["HCSIM_DENYLIST"]).expanduser()
        if not candidate.is_file():
            raise DenylistError(f"HCSIM_DENYLIST points to a missing file: {candidate}")
        return parse_patterns(candidate.read_text())
    if DEFAULT_PATH.is_file():
        return parse_patterns(DEFAULT_PATH.read_text())
    raise DenylistError(
        "no denylist found: pass --patterns, set SANITIZE_DENYLIST or HCSIM_DENYLIST, "
        f"or create {DEFAULT_PATH}"
    )


def scan_text(
    text: str, location: str, patterns: Sequence[Pattern], first_line: int = 1
) -> list[Hit]:
    hits = []
    for offset, line in enumerate(text.splitlines()):
        for pattern in patterns:
            if pattern.regex.search(line):
                hits.append(Hit(location, first_line + offset, pattern.line))
    return hits


def scan_name(name: str, patterns: Sequence[Pattern]) -> list[Hit]:
    hits = [Hit(name, None, p.line, "in the file name") for p in patterns if p.regex.search(name)]
    if any(rule.search(name) for rule in FORBIDDEN_NAMES):
        hits.append(Hit(name, None, None, "private material must never be committed"))
    return hits


def scan_file(path: Path, patterns: Sequence[Pattern], display: str | None = None) -> list[Hit]:
    name = display or path.as_posix()
    hits = scan_name(name, patterns)
    data = path.read_bytes()
    if b"\0" in data[:BINARY_SNIFF_BYTES]:
        return hits
    return hits + scan_text(data.decode("utf-8", errors="replace"), name, patterns)


# --- git ----------------------------------------------------------------------------------------


def git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", "-c", "core.quotePath=false", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise DenylistError(f"git {args[0]} failed: {result.stderr.strip()}")
    return result.stdout


def scan_tree(patterns: Sequence[Pattern], cwd: Path | None = None) -> list[Hit]:
    root = Path(git("rev-parse", "--show-toplevel", cwd=cwd).strip())
    hits = []
    for name in git("ls-files", "-z", cwd=root).split("\0"):
        if name and (root / name).is_file():
            hits += scan_file(root / name, patterns, display=name)
    return hits


def scan_added_lines(diff: str, patterns: Sequence[Pattern]) -> list[Hit]:
    """Scan the added lines of a zero-context unified diff, with new-file line numbers."""
    hits: list[Hit] = []
    current: str | None = None
    line_no = 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            target = line[4:]
            current = None if target == "/dev/null" else target.removeprefix("b/")
        elif line.startswith("@@"):
            if match := re.search(r"\+(\d+)", line):
                line_no = int(match.group(1))
        elif line.startswith("+") and current is not None:
            hits += scan_text(line[1:], current, patterns, first_line=line_no)
            line_no += 1
    return hits


def scan_range(
    base: str, head: str, patterns: Sequence[Pattern], cwd: Path | None = None
) -> list[Hit]:
    """Scan what `head` adds since it diverged from `base`: lines, file names, and messages."""
    since_fork = f"{base}...{head}"
    diff = git("diff", "-U0", "--no-color", "--no-ext-diff", since_fork, cwd=cwd)
    hits = scan_added_lines(diff, patterns)
    names = git("diff", "--name-only", "--diff-filter=AR", "-z", since_fork, cwd=cwd)
    for name in names.split("\0"):
        if name:
            hits += scan_name(name, patterns)
    hits += scan_commit_messages(f"{base}..{head}", patterns, cwd=cwd)
    return hits


def scan_commit_messages(
    revs: str, patterns: Sequence[Pattern], cwd: Path | None = None
) -> list[Hit]:
    hits = []
    for record in git("log", "--format=%H%x00%B%x1e", revs, cwd=cwd).split("\x1e"):
        if "\0" in record:
            sha, message = record.strip("\n").split("\0", 1)
            hits += scan_text(message, f"commit {sha[:12]} message", patterns)
    return hits


def scan_history_names(head: str, cwd: Path | None = None) -> list[Hit]:
    """Every file name in the history reachable from `head`, against the forbidden names."""
    hits = []
    for name in set(git("log", "--name-only", "--format=", head, cwd=cwd).splitlines()):
        if name and any(rule.search(name) for rule in FORBIDDEN_NAMES):
            hits.append(Hit(name, None, None, "private material in the history"))
    return hits


def scan_pre_push(patterns: Sequence[Pattern], env: dict[str, str]) -> list[Hit]:
    """Scan what a push sends, using the refs pre-commit exports for the pre-push stage."""
    head = env.get("PRE_COMMIT_TO_REF", "")
    base = env.get("PRE_COMMIT_FROM_REF", "")
    if not head or ZERO_SHA.match(head):
        return []  # Deleting a remote branch sends nothing.
    hits = scan_history_names(head)
    if not base or ZERO_SHA.match(base):
        base = git("hash-object", "-t", "tree", "/dev/null").strip()  # the empty tree
        hits += scan_added_lines(git("diff", "-U0", "--no-color", base, head), patterns)
        hits += scan_commit_messages(head, patterns)
    else:
        hits += scan_range(base, head, patterns)
    if branch := env.get("PRE_COMMIT_REMOTE_BRANCH", ""):
        hits += scan_text(branch, "remote branch name", patterns)
    return hits


# --- CLI ----------------------------------------------------------------------------------------


def report(hits: Iterable[Hit]) -> int:
    unique = sorted(set(hits), key=str)
    for hit in unique:
        print(hit)
    if unique:
        print(
            f"sanitize-scan: {len(unique)} match(es). Locations only; matched text is never shown."
        )
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan for denylisted patterns.",
        epilog="Exit codes: 0 clean, 1 matches, 2 error.",
    )
    parser.add_argument("--patterns", type=Path, help="denylist file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tree", action="store_true", help="every file git tracks")
    mode.add_argument("--range", metavar="BASE..HEAD", help="what a commit range adds")
    mode.add_argument("--commit-msg", type=Path, metavar="FILE", help="a commit message file")
    mode.add_argument("--stdin", metavar="NAME", help="text on stdin, reported as NAME")
    mode.add_argument("--pre-push", action="store_true", help="what a push sends (pre-commit)")
    parser.add_argument("paths", nargs="*", type=Path, help="files to scan")
    args = parser.parse_args(argv)

    has_mode = any([args.tree, args.range, args.commit_msg, args.stdin, args.pre_push])
    if has_mode == bool(args.paths):
        parser.error("pass files to scan, or one scan mode, but not both")
    try:
        patterns = load_patterns(args.patterns)
        if args.tree:
            hits = scan_tree(patterns)
        elif args.range:
            match = re.fullmatch(r"(.+?)\.{2,3}(.+)", args.range)
            if not match:
                parser.error("--range takes BASE..HEAD")
            hits = scan_range(match.group(1), match.group(2), patterns)
        elif args.commit_msg:
            hits = scan_text(args.commit_msg.read_text(), "commit message", patterns)
        elif args.stdin:
            hits = scan_text(sys.stdin.read(), args.stdin, patterns)
        elif args.pre_push:
            hits = scan_pre_push(patterns, dict(os.environ))
        else:
            hits = [h for path in args.paths if path.is_file() for h in scan_file(path, patterns)]
    except DenylistError as error:
        print(f"sanitize-scan: {error}", file=sys.stderr)
        return 2
    return report(hits)


if __name__ == "__main__":
    sys.exit(main())
