#!/usr/bin/env python3
"""Turn docs/implementation-plan.md into GitHub labels, milestones, issues, and a Project board.

Runs as a dry run by default: it prints a summary and can write a Markdown preview. Pass --apply
to create. Every issue carries a hidden `plan-id` marker, so re-running creates only what's
missing. Re-running with a milestone added to --detail expands that milestone's checklists into
sub-issues.

Before anything is created, every title, body, and description is scanned against the denylist.
A match stops the run, and the report names the location and denylist line, never the matched
text.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = "rio-sanjuan/headcount-simulation"
OWNER = REPO.split("/")[0]
PROJECT_TITLE = "headcount-simulation"
PLAN_URL = f"https://github.com/{REPO}/blob/main/docs/implementation-plan.md"
DEFAULT_PLAN = Path(__file__).resolve().parent.parent / "docs" / "implementation-plan.md"
DEFAULT_DETAIL = "M0,M1"
PACE_SECONDS = 1.5  # Stays under GitHub's content-creation limits (80 a minute, 500 an hour).

AREAS = ["synth", "model", "guidance", "engine", "financial", "studies", "ci", "docs", "app"]

# Looked up by epic id first, then by milestone.
AREA_BY_ID = {
    "M0.1": "ci", "M0.2": "ci", "M0.3": "ci", "M0.4": "docs",
    "M1.1": "synth", "M1.2": "model", "M1.3": "model", "M1.4": "model", "M1.5": "guidance",
    "M1.6": "engine", "M1.7": "financial", "M1.8": "studies", "M1.9": "ci",
    "M2": "synth", "M3": "model", "M4": "engine", "M5": "guidance", "M6": "financial",
    "M7": "engine", "M8": "studies",
    "M9.1": "engine", "M9.2": "app", "M9.3": "docs", "M9.4": "ci",
}

# Critical-path "blocked by" links between epics. Milestone-level links come from the plan's
# mermaid graph and connect the acceptance issues.
EPIC_DEPS = {
    "M0.2": ["M0.1"], "M0.3": ["M0.1"], "M0.4": ["M0.1"],
    "M1.1": ["M0.1"], "M1.2": ["M0.1"],
    "M1.3": ["M1.2"], "M1.4": ["M1.2"], "M1.5": ["M1.1", "M1.2"],
    "M1.6": ["M1.2", "M1.3", "M1.4"], "M1.7": ["M1.2"],
    "M1.8": ["M1.5", "M1.6", "M1.7"], "M1.9": ["M1.8", "M0.2"],
}

# Hand-written titles for tasks whose first sentence doesn't make a good title. Keys are plan ids.
TITLES = {
    "M0.1/1": "Add NOTICE, README, CONTRIBUTING, and SECURITY",
    "M0.1/3": "Add the `stat_check` helper for statistical tests",
    "M0.1/4": "Configure `ruff`, `mypy`, `pytest`, `hypothesis`, and `pytest-xdist`",
    "M0.1/5": "Add `pre-commit` and `.gitignore` entries for generated output",
    "M0.2/1": "Add `make check` and `make check-all`",
    "M0.2/2": "Add `ci.yml`: fast and heavy jobs, test selection, and `ci-pass`",
    "M0.2/3": "Add `post-merge.yml` for macOS runs on `main`",
    "M0.2/4": "Add quality gates: diff coverage, peak memory, and metric baselines",
    "M0.2/6": "Configure Dependabot with grouped weekly updates",
    "M0.2/7": "Harden CI permissions and protect `main`",
    "M0.2/8": "Add issue and PR templates",
    "M0.2/9": "Set the contribution policy: issues yes, PRs no",
    "M0.3/7": "Scan issue and PR text, branch names, tags, and release notes",
    "M0.4/1": "Write `docs/design/overview.md`",
    "M0.4/2": "Write `docs/design/notation.md` and `docs/glossary.md`",
    "M0.4/3": "Write ADRs 0001–0012",
    "M1.1/2": "Build the lite monthly workforce process",
    "M1.1/3": "Build the lite requisition process",
    "M1.1/4": "Add lite compensation ranges, stock price, awards, and reference ledger",
    "M1.1/5": "Add the `hcsim synth generate` command",
    "M1.2/1": "Write `SimulationSchema`",
    "M1.2/5": "Write the `GuidanceForecast` artifact",
    "M1.2/6": "Write the core model protocols",
    "M1.3/1": "Write the Bernoulli and quota samplers",
    "M1.4/2": "Write the backfill rate, growth count, and attribute generator",
    "M1.5/1": "Ingest and aggregate guidance inputs from DuckDB",
    "M1.5/2": "Fit and select between two benchmark models",
    "M1.5/3": "Reconcile and write the guidance artifact",
    "M1.6/1": "Write `SimulationConfig`, `SimulationState`, and the seed manager",
    "M1.6/3": "Write `DomainResult` and the apply step",
    "M1.6/4": "Build the step loop with five domains",
    "M1.7/1": "Write the new-hire strategy, a vest template, and the RSU tracker",
    "M1.7/3": "Write the Phase 3 expense calculator and SBC output",
    "M1.8/1": "Write the minimal profile, one-arm harness, and `invariants` evaluator",
    "M1.9/1": "Run the e2e skeleton and determinism check in CI",
}

LABELS = {
    "epic": ("3E4B9E", "Parent issue for one plan epic"),
    "task": ("0E8A16", "One PR-sized unit of work"),
    "acceptance": ("FBCA04", "A milestone's definition of done"),
    "question": ("D876E3", "An open design question"),
    "future": ("C5DEF5", "Deferred work outside every milestone"),
    "cloud": ("BFD4F2", "Deferred cloud infrastructure"),
    **{f"area:{a}": ("1D76DB", f"Area: {a}") for a in AREAS},
}

DONE_TASK = "**Done when:** it's merged as one PR, with tests where they apply."
DONE_EPIC = (
    "### Done when\n"
    "Every task is done. The milestone's own criteria live in its acceptance issue."
)


@dataclass
class Task:
    text: str
    sub: list[str] = field(default_factory=list)


@dataclass
class Epic:
    key: str
    name: str
    note: str = ""
    lead: list[str] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)


@dataclass
class MilestonePlan:
    key: str
    title: str
    short: str = ""
    outcome: str = ""
    acceptance: str = ""
    intro: list[str] = field(default_factory=list)
    epics: list[Epic] = field(default_factory=list)


@dataclass
class Issue:
    key: str
    title: str
    body: str
    labels: list[str]
    milestone: str | None
    parent: str | None = None
    blocked_by: list[str] = field(default_factory=list)


# --- Parsing -------------------------------------------------------------------------------------


def section(lines: list[str], number: str) -> list[str]:
    start = next(i for i, line in enumerate(lines) if line.startswith(f"## {number}. "))
    end = next(
        (i for i in range(start + 1, len(lines)) if re.match(r"^## \d+\. ", lines[i])),
        len(lines),
    )
    return lines[start + 1 : end]


def parse_plan(text: str) -> tuple[list[MilestonePlan], list[tuple[str, str]], list[str], list[dict]]:
    lines = text.splitlines()
    milestones: dict[str, MilestonePlan] = {}

    for line in section(lines, "5"):
        if m := re.match(r"^### (M\d) — (.+)$", line):
            milestones[m.group(1)] = MilestonePlan(m.group(1), f"{m.group(1)} — {m.group(2)}")
    current_m: MilestonePlan | None = None
    epic: Epic | None = None
    for line in section(lines, "5"):
        if m := re.match(r"^### (M\d) — ", line):
            current_m, epic = milestones[m.group(1)], None
        elif m := re.match(r"^\*\*(M\d\.\d+) (.+?)\*\*\s*(.*)$", line):
            note = m.group(3).strip()
            if nm := re.match(r"^\((continues in .+)\)$", note):
                note = nm.group(1)[0].upper() + nm.group(1)[1:] + "."
            epic = Epic(m.group(1), m.group(2), note)
            assert current_m is not None
            current_m.epics.append(epic)
        elif m := re.match(r"^- (.*)$", line):
            assert epic is not None, line
            epic.tasks.append(Task(m.group(1)))
        elif m := re.match(r"^  (\s*\S.*)$", line):
            assert epic is not None and epic.tasks, line
            epic.tasks[-1].sub.append(m.group(1))
        elif line.strip():
            if epic is not None and not epic.tasks:
                epic.lead.append(line.strip())
            elif epic is None and current_m is not None:
                current_m.intro.append(line.strip())

    edges: list[tuple[str, str]] = []
    for line in section(lines, "4"):
        if m := re.match(r"^\| (M\d) ([^|]+?) \| ([^|]+?) \| ([^|]+?) \|$", line):
            ms = milestones[m.group(1)]
            ms.short, ms.outcome, ms.acceptance = m.group(2), m.group(3), m.group(4)
        elif m := re.match(r"^\s*(M\d)(?:\[[^\]]*\])?\s*-->\s*(M\d)", line):
            edges.append((m.group(1), m.group(2)))

    future = [m.group(1) for line in section(lines, "9") if (m := re.match(r"^- (.*)$", line))]

    questions = []
    for line in section(lines, "7"):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[1].startswith("Open"):
            questions.append({"question": cells[0], "resolution": cells[1], "epic": cells[2]})

    return list(milestones.values()), edges, future, questions


# --- Building issues -----------------------------------------------------------------------------


def short_title(text: str, limit: int = 90) -> str:
    first = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0].rstrip(".:")
    if (m := re.match(r"^([^:`]{3,60}):\s", first)) and len(m.group(1).split()) >= 2:
        first = m.group(1)
    if len(first) > limit:
        cut = first[:limit]
        for sep in (" (", ", ", " with ", " and "):
            if (i := cut.rfind(sep)) >= 40:
                first = first[:i]
                break
        else:
            first = cut.rsplit(" ", 1)[0] + "…"
    # Never leave an unclosed parenthesis or code span.
    if first.count("(") > first.count(")"):
        first = first[: first.rfind(" (")]
    if first.count("`") % 2:
        first = first[: first.rfind("`")].rstrip(" ,")
    return first.rstrip(" ,:")


def slug(heading: str) -> str:
    s = re.sub(r"[^\w\- ]", "", heading.lower())
    return s.replace(" ", "-")


def area_of(key: str) -> str:
    return AREA_BY_ID.get(key) or AREA_BY_ID[key.split(".")[0]]


def task_markdown(task: Task, checkbox: bool) -> str:
    head = f"- [ ] {task.text}" if checkbox else task.text
    indent = "  " if checkbox else ""
    return "\n".join([head, *(indent + s for s in task.sub)])


def build(milestones: list[MilestonePlan], edges, future, questions, detail: set[str]):
    issues: list[Issue] = []
    for ms in milestones:
        acc = [s for s in re.split(r"(?<=\.)\s+(?=[A-Z`])", ms.acceptance) if s]
        issues.append(Issue(
            key=f"{ms.key}/acceptance",
            title=f"{ms.key} acceptance: {ms.short}",
            body="\n\n".join([
                f"<!-- plan-id: {ms.key}/acceptance -->",
                f"**Outcome:** {ms.outcome}",
                "### Criteria\n" + "\n".join(f"- [ ] {s}" for s in acc),
                "Close this issue last in the milestone, once every criterion holds.",
            ]),
            labels=["acceptance"],
            milestone=ms.key,
            blocked_by=[f"{a}/acceptance" for a, b in edges if b == ms.key],
        ))
        detailed = ms.key in detail
        for ep in ms.epics:
            parts = [f"<!-- plan-id: {ep.key} -->"]
            if ep.lead:
                parts.append(" ".join(ep.lead))
            if ep.note:
                parts.append(ep.note)
            parts.append(f"**Milestone:** [{ms.title}]({PLAN_URL}#{slug(ms.title)})")
            if detailed:
                parts.append("### Tasks\n<!-- tasks: sub-issues -->\nTracked as sub-issues of this epic.")
            else:
                checklist = "\n".join(task_markdown(t, checkbox=True) for t in ep.tasks)
                parts.append(
                    "### Tasks\n<!-- tasks: checklist -->\n" + checklist
                    + "\n\nThese become sub-issues when this milestone is next."
                )
            parts.append(DONE_EPIC)
            issues.append(Issue(
                key=ep.key,
                title=f"{ep.key} {ep.name}",
                body="\n\n".join(parts),
                labels=["epic", f"area:{area_of(ep.key)}"],
                milestone=ms.key,
                blocked_by=list(EPIC_DEPS.get(ep.key, [])),
            ))
            if detailed:
                for i, t in enumerate(ep.tasks, 1):
                    issues.append(Issue(
                        key=f"{ep.key}/{i}",
                        title=f"{ep.key} · {TITLES.get(f'{ep.key}/{i}') or short_title(t.text)}",
                        body="\n\n".join([
                            f"<!-- plan-id: {ep.key}/{i} -->",
                            task_markdown(t, checkbox=False),
                            DONE_TASK,
                        ]),
                        labels=["task", f"area:{area_of(ep.key)}"],
                        milestone=ms.key,
                        parent=ep.key,
                    ))
    for i, q in enumerate(questions, 1):
        issues.append(Issue(
            key=f"{q['epic']}/q{i}",
            title=f"{q['epic']} · Question: {q['question']}",
            body="\n\n".join([
                f"<!-- plan-id: {q['epic']}/q{i} -->",
                f"Open design question from section 7.1 of the plan: {q['question'].lower()}.",
                f"Resolve it before {q['epic']} closes.",
            ]),
            labels=["question", f"area:{area_of(q['epic'])}"],
            milestone=q["epic"].split(".")[0],
            parent=q["epic"],
        ))
    for i, text in enumerate(future, 1):
        issues.append(Issue(
            key=f"future/{i}",
            title=short_title(text),
            body="\n\n".join([
                f"<!-- plan-id: future/{i} -->",
                text,
                "Deferred: cloud infrastructure is outside milestones M0–M9 (section 9 of the plan).",
            ]),
            labels=["future", "cloud"],
            milestone=None,
        ))
    return issues


# --- Sanitization --------------------------------------------------------------------------------


def load_denylist(path: Path) -> list[tuple[int, re.Pattern[str]]]:
    patterns = []
    for n, raw in enumerate(path.read_text().splitlines(), 1):
        line = re.sub(r"\s+#.*$", "", raw).strip()
        if line and not line.startswith("#"):
            patterns.append((n, re.compile(line, re.IGNORECASE)))
    return patterns


def scan(named_texts: list[tuple[str, str]], patterns) -> list[str]:
    hits = []
    for name, text in named_texts:
        for ln, line in enumerate(text.splitlines(), 1):
            for n, pat in patterns:
                if pat.search(line):
                    hits.append(f"{name}, line {ln}: denylist line {n}")
    return hits


# --- GitHub --------------------------------------------------------------------------------------


def gh(*args: str, stdin: str | None = None) -> str:
    for attempt in range(1, 7):
        p = subprocess.run(["gh", *args], input=stdin, capture_output=True, text=True)
        if p.returncode == 0:
            return p.stdout
        if re.search(r"rate limit|secondary|abuse", p.stderr, re.IGNORECASE):
            wait = 60 * attempt
            print(f"  rate limited; waiting {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        sys.exit(f"gh {' '.join(args[:3])} failed: {p.stderr.strip()}")
    sys.exit("gave up after repeated rate limiting")


def gh_lines(*args: str) -> list[dict]:
    out = gh(*args)
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def create(*args: str, stdin: str | None = None) -> str:
    out = gh(*args, stdin=stdin)
    time.sleep(PACE_SECONDS)
    return out


def apply(milestones, issues: list[Issue], detail: set[str]) -> None:
    existing_labels = {d["name"] for d in gh_lines(
        "api", f"repos/{REPO}/labels", "--paginate", "--jq", ".[] | {name} | tojson")}
    for name, (color, desc) in LABELS.items():
        if name not in existing_labels:
            create("api", f"repos/{REPO}/labels", "--method", "POST",
                   "-f", f"name={name}", "-f", f"color={color}", "-f", f"description={desc}")
            print(f"label   {name}")

    ms_numbers = {d["title"]: d["number"] for d in gh_lines(
        "api", f"repos/{REPO}/milestones?state=all", "--paginate",
        "--jq", ".[] | {title, number} | tojson")}
    ms_by_key = {}
    for ms in milestones:
        if ms.title not in ms_numbers:
            out = create("api", f"repos/{REPO}/milestones", "--method", "POST",
                         "-f", f"title={ms.title}", "-f", f"description={ms.outcome}")
            ms_numbers[ms.title] = json.loads(out)["number"]
            print(f"milestone {ms.title}")
        ms_by_key[ms.key] = ms_numbers[ms.title]

    found: dict[str, dict] = {}
    for d in gh_lines("api", f"repos/{REPO}/issues?state=all&per_page=100", "--paginate",
                      "--jq", '.[] | {number, id, url: .html_url, body: (.body // ""), '
                              'pr: has("pull_request")} | tojson'):
        if not d["pr"] and (m := re.search(r"<!-- plan-id: (\S+) -->", d["body"])):
            found[m.group(1)] = d

    for issue in issues:
        if issue.key in found:
            body = found[issue.key]["body"]
            if "<!-- tasks: checklist -->" in body and issue.key.split(".")[0] in detail:
                create("api", f"repos/{REPO}/issues/{found[issue.key]['number']}",
                       "--method", "PATCH", "--input", "-",
                       stdin=json.dumps({"body": issue.body}))
                print(f"update  #{found[issue.key]['number']} {issue.title}")
            continue
        payload = {"title": issue.title, "body": issue.body, "labels": issue.labels}
        if issue.milestone:
            payload["milestone"] = ms_by_key[issue.milestone]
        out = json.loads(create("api", f"repos/{REPO}/issues", "--method", "POST",
                                "--input", "-", stdin=json.dumps(payload)))
        found[issue.key] = {"number": out["number"], "id": out["id"], "url": out["html_url"]}
        print(f"issue   #{out['number']} {issue.title}")

    for issue in issues:
        if issue.parent:
            parent = found[issue.parent]["number"]
            have = {d["id"] for d in gh_lines(
                "api", f"repos/{REPO}/issues/{parent}/sub_issues", "--paginate",
                "--jq", ".[] | {id} | tojson")}
            if found[issue.key]["id"] not in have:
                create("api", f"repos/{REPO}/issues/{parent}/sub_issues", "--method", "POST",
                       "-F", f"sub_issue_id={found[issue.key]['id']}")
                print(f"sub     #{found[issue.key]['number']} -> #{parent}")
        if issue.blocked_by:
            number = found[issue.key]["number"]
            have = {d["id"] for d in gh_lines(
                "api", f"repos/{REPO}/issues/{number}/dependencies/blocked_by", "--paginate",
                "--jq", ".[] | {id} | tojson")}
            for blocker in issue.blocked_by:
                if found[blocker]["id"] not in have:
                    create("api", f"repos/{REPO}/issues/{number}/dependencies/blocked_by",
                           "--method", "POST", "-F", f"issue_id={found[blocker]['id']}")
                    print(f"blocks  #{found[blocker]['number']} blocks #{number}")

    projects = json.loads(gh("project", "list", "--owner", OWNER, "--format", "json"))["projects"]
    project = next((p for p in projects if p["title"] == PROJECT_TITLE), None)
    if project is None:
        project = json.loads(create("project", "create", "--owner", OWNER,
                                    "--title", PROJECT_TITLE, "--format", "json"))
        print(f"project #{project['number']} {PROJECT_TITLE}")
    number = str(project["number"])
    # Both steps are safe to repeat, so a resumed run finishes them too.
    create("project", "edit", number, "--owner", OWNER, "--visibility", "PUBLIC")
    p = subprocess.run(["gh", "project", "link", number, "--owner", OWNER,
                        "--repo", REPO.split("/")[1]], capture_output=True, text=True)
    if p.returncode != 0 and "already" not in p.stderr.lower():
        print(f"  warning: linking the project to the repo failed: {p.stderr.strip()}",
              file=sys.stderr)
    items = json.loads(gh("project", "item-list", number, "--owner", OWNER,
                          "--format", "json", "--limit", "2000"))["items"]
    in_project = {(i.get("content") or {}).get("url") for i in items}
    for issue in issues:
        url = found[issue.key]["url"]
        if url not in in_project:
            create("project", "item-add", number, "--owner", OWNER, "--url", url)
    print("done")


# --- Main ----------------------------------------------------------------------------------------


def preview(milestones, issues: list[Issue]) -> str:
    out = ["# Issue preview", ""]
    by_key = {i.key: i for i in issues}
    for ms in milestones:
        out += [f"## Milestone: {ms.title}", "", f"*Description:* {ms.outcome}", ""]
        for issue in issues:
            if issue.milestone == ms.key:
                out += [f"### {issue.title}", "",
                        f"*Labels:* {', '.join(issue.labels)}"]
                if issue.parent:
                    out.append(f"*Parent:* {by_key[issue.parent].title}")
                if issue.blocked_by:
                    out.append("*Blocked by:* " + "; ".join(by_key[b].title for b in issue.blocked_by))
                out += ["", "```markdown", issue.body, "```", ""]
    out += ["## No milestone", ""]
    for issue in issues:
        if issue.milestone is None:
            out += [f"### {issue.title}", "", f"*Labels:* {', '.join(issue.labels)}", "",
                    "```markdown", issue.body, "```", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    ap.add_argument("--denylist", type=Path,
                    default=Path(os.environ.get("HCSIM_DENYLIST", "~/.config/hcsim/denylist.txt")).expanduser())
    ap.add_argument("--detail", default=DEFAULT_DETAIL,
                    help="milestones whose tasks become sub-issues (default: %(default)s)")
    ap.add_argument("--preview", type=Path, help="write a Markdown preview of every issue here")
    ap.add_argument("--apply", action="store_true", help="create on GitHub (default: dry run)")
    args = ap.parse_args()

    detail = {s.strip() for s in args.detail.split(",") if s.strip()}
    milestones, edges, future, questions = parse_plan(args.plan.read_text())
    issues = build(milestones, edges, future, questions, detail)
    keys = {i.key for i in issues}
    stale = [k for k in TITLES if k.split(".")[0] in detail and k not in keys]
    if stale:
        sys.exit(f"title overrides with no matching task (did the plan change?): {stale}")

    texts = [(f"label {k}", d) for k, (_, d) in LABELS.items()]
    texts += [(f"milestone {m.key}", f"{m.title}\n{m.outcome}") for m in milestones]
    texts += [(f"{i.key} title", i.title) for i in issues]
    texts += [(f"{i.key} body", i.body) for i in issues]
    hits = scan(texts, load_denylist(args.denylist))

    counts: dict[str, int] = {}
    for i in issues:
        kind = i.labels[0]
        counts[kind] = counts.get(kind, 0) + 1
    links = sum(1 for i in issues if i.parent)
    deps = sum(len(i.blocked_by) for i in issues)
    print(f"milestones: {len(milestones)}   issues: {len(issues)}  "
          + "  ".join(f"{k}: {v}" for k, v in counts.items()))
    print(f"sub-issue links: {links}   blocked-by links: {deps}   labels: {len(LABELS)}")
    print(f"content-creating calls, worst case: "
          f"{len(LABELS) + len(milestones) + len(issues) + links + deps + 3 + len(issues)}")
    long_titles = [i.title for i in issues if len(i.title) > 100]
    if long_titles:
        print(f"titles over 100 characters: {len(long_titles)}")

    if args.preview:
        args.preview.write_text(preview(milestones, issues))
        print(f"preview written to {args.preview}")
    if hits:
        print(f"\nDENYLIST: {len(hits)} match(es). Nothing will be created.")
        for h in hits:
            print(f"  {h}")
        sys.exit(1)
    print("denylist: clean")
    if args.apply:
        apply(milestones, issues, detail)
    else:
        print("dry run: nothing created (pass --apply to create)")


if __name__ == "__main__":
    main()
