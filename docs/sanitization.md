# Sanitization policy

This repository is public, and all of its data is synthetic. This page states what may never appear in it and how that's enforced. It covers code, docs, notebooks, commit messages, branch and tag names, issues, pull requests, comments, releases, and CI artifacts.

## The rules

1. **No real-world specifics.** No company, product, or internal system names. No internal codes, organization or cost-center identifiers, or names of real people or teams.
2. **No measured figures.** Every default parameter is invented, round, and documented as synthetic. No default is copied from a real measurement, even a re-derived one.
3. **Generic vocabulary.** Concepts use the public names in the [glossary](glossary.md), such as `org_level{n}`, `cost_center`, `TERM`, `XFER_IN`, `regular` and `contingent` accounts, and rating tiers `R1`–`R5`.
4. **Synthetic data only.** Every input comes from the seeded generator in `hcsim.synth`. Examples, tests, and fixtures use obviously synthetic values.

## How it's enforced

A denylist scanner, [`tools/sanitize_scan.py`](https://github.com/rio-sanjuan/headcount-simulation/blob/main/tools/sanitize_scan.py), checks text against a list of patterns. **The list itself is private.** It never enters the repository: a maintainer keeps it in a local file, and CI reads it from an encrypted secret. Publishing it would disclose exactly what it protects.

The scanner runs at every point where text becomes public.

| Where | What it scans |
|---|---|
| `pre-commit` hook | Staged files, content and names |
| `commit-msg` hook | The commit message |
| `pre-push` hook | What the push sends: added lines, file names, commit messages, the branch name, and private file names anywhere in the history |
| CI, every PR | What the PR adds since its base, and its branch name |
| CI, every push to `main` | Every tracked file, and the pushed commits |
| CI, text events | Issue and PR titles and bodies, comments, branch and tag names, and release notes |
| CI, before any artifact upload | Every file in the artifact |

Three properties make the scanner safe to run in public:

- **It never shows what matched.** Reports name a location and a denylist line number only, such as `docs/x.md:12: denylist line 7`, because CI logs on a public repository are public.
- **It fails closed.** With no list available it exits with an error rather than passing.
- **It proves itself.** The list includes a harmless canary pattern, so a test can plant the canary and confirm that each scan fires, without publishing a real term.

Notebooks are committed with their outputs stripped, so no rendered result can carry data in, and their JSON is scanned as text. A separate `gitleaks` job scans the history for credentials. The wiki and Discussions are off, because nothing scans them.

## When the scanner matches

- **Fix the text.** Don't bypass the hook with `--no-verify`: the pre-push hook and CI scan the same commits again.
- **A false positive** means the pattern is too broad. The maintainer tightens it in the private list, for example with word boundaries.
- **An issue or comment that matches** gets the `needs-sanitization` label. Edit it before anything else.

## If something sensitive was published

Anything pushed to a public repository should be treated as disclosed, because it may already be cached or mirrored.

1. Remove it at once: edit or delete the issue or comment, or revert the change.
2. Report it privately as described in [SECURITY.md](https://github.com/rio-sanjuan/headcount-simulation/blob/main/SECURITY.md), so the maintainer can judge whether history needs rewriting and whether GitHub should purge cached views.
3. Add a pattern to the private list, so the same text can't reappear.
