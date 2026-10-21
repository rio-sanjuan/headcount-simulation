# Contributing

Thanks for your interest in this project.

## Issues are welcome

Please open an issue for:

- **Bugs.** Include the command you ran, what you expected, what happened, and your OS and Python version.
- **Questions** about the design or the code.
- **Ideas** for improvements.

Check the [open issues](https://github.com/rio-sanjuan/headcount-simulation/issues) and the [implementation plan](docs/implementation-plan.md) first, since the work ahead is tracked there.

## Pull requests aren't accepted

This project doesn't accept pull requests from outside contributors, and a workflow closes them automatically with a pointer to this file. If you'd like to suggest a change, please open an issue describing it instead.

## Content rules

Everything in this repository is public and synthetic. Issues and comments follow the same rules as the code:

- No company, product, or internal system names, internal codes, or org identifiers.
- No real people's names, and no real workforce, compensation, or financial figures.
- Examples use synthetic values only.

If you notice content that looks like real or proprietary data, please report it privately as described in [SECURITY.md](SECURITY.md) rather than in a public issue.

## How the maintainer works

These conventions apply to work on the repository itself.

- **One task, one PR.** Each task in the plan is a GitHub issue, and each PR closes one task. PRs are squash-merged.
- **Conventional commits.** Commit messages and PR titles follow [Conventional Commits](https://www.conventionalcommits.org/), such as `feat:`, `fix:`, `docs:`, `build:`, `ci:`, and `test:`.
- **Local checks.** `pre-commit` runs on every commit, and the full suite runs in CI on every PR.
- **Sanitization.** A denylist scanner checks files, commit messages, and issue and PR text before anything is published. Its patterns are kept private.
