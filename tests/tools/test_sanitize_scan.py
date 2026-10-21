"""Tests for the sanitization scanner.

Every pattern and token here is a made-up canary, never an entry from the real denylist.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.sanitize_scan import (
    DenylistError,
    Hit,
    load_patterns,
    parse_patterns,
    scan_added_lines,
    scan_file,
    scan_name,
    scan_pre_push,
    scan_range,
    scan_text,
    scan_tree,
)

ROOT = Path(__file__).resolve().parents[2]
DENYLIST = """\
# canary patterns for tests
zebra-canary-\\d+
quokka   # inline comment
\\bwombat\\b
"""
TOKEN = "zebra-canary-42"


@pytest.fixture
def patterns_file(tmp_path: Path) -> Path:
    path = tmp_path / "patterns.txt"
    path.write_text(DENYLIST)
    return path


def run_cli(
    *args: str, env: dict[str, str] | None = None, stdin: str | None = None, cwd: Path = ROOT
) -> subprocess.CompletedProcess[str]:
    full_env = {
        k: v for k, v in os.environ.items() if k not in {"SANITIZE_DENYLIST", "HCSIM_DENYLIST"}
    }
    full_env["HOME"] = str(cwd)  # so the default ~/.config path is absent
    full_env.update(env or {})
    return subprocess.run(
        [sys.executable, str(ROOT / "tools" / "sanitize_scan.py"), *args],
        capture_output=True,
        text=True,
        env=full_env,
        input=stdin,
        cwd=cwd,
        check=False,
    )


def assert_never_revealed(result: subprocess.CompletedProcess[str], *secrets: str) -> None:
    for secret in secrets:
        assert secret not in result.stdout
        assert secret not in result.stderr


# --- patterns -----------------------------------------------------------------------------------


def test_comments_and_blank_lines_are_skipped_and_line_numbers_kept() -> None:
    patterns = parse_patterns(DENYLIST)
    assert [p.line for p in patterns] == [2, 3, 4]
    assert patterns[1].regex.pattern == "quokka"


def test_an_invalid_pattern_is_refused_without_echoing_it() -> None:
    with pytest.raises(DenylistError) as error:
        parse_patterns("ok\nsecret-(unclosed\n")
    assert "line 2" in str(error.value)
    assert "secret" not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__


def test_an_empty_denylist_is_refused() -> None:
    with pytest.raises(DenylistError, match="no patterns"):
        parse_patterns("# only a comment\n\n")


def test_patterns_load_from_the_environment_in_priority_order(patterns_file: Path) -> None:
    assert len(load_patterns(env={"SANITIZE_DENYLIST": DENYLIST})) == 3
    assert len(load_patterns(env={"HCSIM_DENYLIST": str(patterns_file)})) == 3
    with pytest.raises(DenylistError, match="missing file"):
        load_patterns(env={"HCSIM_DENYLIST": str(patterns_file) + ".gone"})


# --- text and files -----------------------------------------------------------------------------


def test_matches_are_case_insensitive_and_located() -> None:
    patterns = parse_patterns(DENYLIST)
    hits = scan_text("fine\nA QUOKKA here\nand zebra-canary-7", "doc", patterns)
    assert [(h.line, h.pattern_line) for h in hits] == [(2, 3), (3, 2)]
    assert str(hits[0]) == "doc:2: denylist line 3"


def test_word_boundaries_avoid_false_positives() -> None:
    patterns = parse_patterns(DENYLIST)
    assert scan_text("wombats and wombatlike", "doc", patterns) == []
    assert scan_text("a wombat", "doc", patterns)


def test_binary_files_are_skipped_but_their_names_are_scanned(tmp_path: Path) -> None:
    patterns = parse_patterns(DENYLIST)
    binary = tmp_path / "quokka.bin"
    binary.write_bytes(b"\0\1" + TOKEN.encode())
    hits = scan_file(binary, patterns, display="quokka.bin")
    assert hits == [Hit("quokka.bin", None, 3, "in the file name")]


@pytest.mark.parametrize(
    "name",
    [
        "project-description.md",
        "docs/project-description.md",
        "my-denylist.txt",
        "denylist.proposed.txt",
    ],
)
def test_private_file_names_are_forbidden(name: str) -> None:
    hits = scan_name(name, parse_patterns(DENYLIST))
    assert any(h.pattern_line is None for h in hits)


def test_added_lines_carry_new_file_line_numbers() -> None:
    diff = (
        "diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n"
        "@@ -1,0 +5,2 @@\n+ok\n+zebra-canary-1\n"
        "diff --git a/gone.txt b/gone.txt\n--- a/gone.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n"
        "-quokka\n"
    )
    hits = scan_added_lines(diff, parse_patterns(DENYLIST))
    assert hits == [Hit("a.txt", 6, 2)]


# --- git modes ----------------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for key, value in {
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
    }.items():
        monkeypatch.setenv(key, value)
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    (root / "README.md").write_text("clean\n")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "initial")
    return root


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout


def commit_file(root: Path, name: str, text: str, message: str) -> str:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    git(root, "add", name)
    git(root, "commit", "-q", "-m", message)
    return git(root, "rev-parse", "HEAD").strip()


def test_range_scans_added_lines_names_and_messages(repo: Path) -> None:
    base = git(repo, "rev-parse", "HEAD").strip()
    git(repo, "checkout", "-q", "-b", "feature")
    commit_file(repo, "notes/quokka.md", "line one\nzebra-canary-9\n", "add notes")
    commit_file(repo, "other.md", "fine\n", "mentions a wombat")
    hits = scan_range(base, "HEAD", parse_patterns(DENYLIST), cwd=repo)
    rendered = sorted(str(h) for h in hits)
    assert "notes/quokka.md:2: denylist line 2" in rendered
    assert "notes/quokka.md: denylist line 3 (in the file name)" in rendered
    assert any(
        r.startswith("commit ") and r.endswith("message:1: denylist line 4") for r in rendered
    )


def test_range_ignores_changes_on_the_base_branch(repo: Path) -> None:
    git(repo, "checkout", "-q", "-b", "feature")
    commit_file(repo, "feature.md", "fine\n", "feature work")
    git(repo, "checkout", "-q", "main")
    commit_file(repo, "main.md", "zebra-canary-3\n", "main work")
    assert scan_range("main", "feature", parse_patterns(DENYLIST), cwd=repo) == []


def test_tree_scans_every_tracked_file(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commit_file(repo, "a/b.txt", "x\nquokka\n", "add")
    (repo / "untracked.txt").write_text("quokka\n")
    monkeypatch.chdir(repo)
    hits = scan_tree(parse_patterns(DENYLIST))
    assert [str(h) for h in hits] == ["a/b.txt:2: denylist line 3"]


def test_pre_push_refuses_private_material_anywhere_in_history(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit_file(repo, "project-description.md", "private\n", "oops")
    git(repo, "rm", "-q", "project-description.md")
    git(repo, "commit", "-q", "-m", "remove it again")
    head = git(repo, "rev-parse", "HEAD").strip()
    monkeypatch.chdir(repo)
    env = {
        "PRE_COMMIT_FROM_REF": "0" * 40,
        "PRE_COMMIT_TO_REF": head,
        "PRE_COMMIT_REMOTE_BRANCH": "refs/heads/main",
    }
    hits = scan_pre_push(parse_patterns(DENYLIST), env)
    assert any("private material in the history" in str(h) for h in hits)


def test_pre_push_scans_the_pushed_range_and_branch_name(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = git(repo, "rev-parse", "HEAD").strip()
    head = commit_file(repo, "x.md", "zebra-canary-5\n", "add x")
    monkeypatch.chdir(repo)
    env = {
        "PRE_COMMIT_FROM_REF": base,
        "PRE_COMMIT_TO_REF": head,
        "PRE_COMMIT_REMOTE_BRANCH": "refs/heads/quokka-work",
    }
    rendered = {str(h) for h in scan_pre_push(parse_patterns(DENYLIST), env)}
    assert rendered == {"x.md:1: denylist line 2", "remote branch name:1: denylist line 3"}


def test_deleting_a_remote_branch_sends_nothing_to_scan() -> None:
    env = {"PRE_COMMIT_FROM_REF": "abc", "PRE_COMMIT_TO_REF": "0" * 40}
    assert scan_pre_push(parse_patterns(DENYLIST), env) == []


# --- CLI ----------------------------------------------------------------------------------------


def test_cli_exits_1_on_a_match_and_never_prints_the_match(
    patterns_file: Path, tmp_path: Path
) -> None:
    target = tmp_path / "doc.md"
    target.write_text(f"hello\n{TOKEN}\n")
    result = run_cli("--patterns", str(patterns_file), str(target))
    assert result.returncode == 1
    assert "doc.md:2: denylist line 2" in result.stdout
    assert_never_revealed(result, TOKEN, "zebra-canary")


def test_cli_reads_the_list_from_the_secret_variable(tmp_path: Path) -> None:
    result = run_cli(
        "--stdin",
        "issue body",
        env={"SANITIZE_DENYLIST": DENYLIST},
        stdin=f"text with {TOKEN}",
        cwd=tmp_path,
    )
    assert result.returncode == 1
    assert result.stdout.startswith("issue body:1: denylist line 2")
    assert_never_revealed(result, TOKEN)


def test_cli_exits_0_when_clean(patterns_file: Path, tmp_path: Path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("nothing to see\n")
    result = run_cli("--patterns", str(patterns_file), str(target))
    assert (result.returncode, result.stdout) == (0, "")


def test_cli_fails_closed_without_a_denylist(tmp_path: Path) -> None:
    result = run_cli("--stdin", "x", stdin="anything", cwd=tmp_path)
    assert result.returncode == 2
    assert "no denylist found" in result.stderr


def test_cli_scans_a_commit_message_file(patterns_file: Path, tmp_path: Path) -> None:
    message = tmp_path / "COMMIT_EDITMSG"
    message.write_text("feat: add the quokka module\n")
    result = run_cli("--patterns", str(patterns_file), "--commit-msg", str(message))
    assert result.returncode == 1
    assert "commit message:1: denylist line 3" in result.stdout


def test_cli_requires_files_or_exactly_one_mode(patterns_file: Path) -> None:
    assert run_cli("--patterns", str(patterns_file)).returncode == 2
    assert run_cli("--patterns", str(patterns_file), "--tree", "README.md").returncode == 2


def test_pre_push_without_refs_scans_all_of_head(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit_file(repo, "y.md", "fine\nquokka\n", "add y")
    monkeypatch.chdir(repo)
    rendered = {str(h) for h in scan_pre_push(parse_patterns(DENYLIST), {})}
    assert rendered == {"y.md:2: denylist line 3"}


@pytest.mark.parametrize("make_path", ["missing", "directory"])
def test_cli_refuses_a_path_that_isnt_a_file(
    make_path: str, patterns_file: Path, tmp_path: Path
) -> None:
    target = tmp_path / make_path
    if make_path == "directory":
        target.mkdir()
    result = run_cli("--patterns", str(patterns_file), str(target))
    assert result.returncode == 2
    assert "not a file" in result.stderr
