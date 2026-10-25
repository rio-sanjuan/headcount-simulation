"""The `hcsim` command. Each subcommand lives in its own module and loads only when called.

Exit codes: 0 success, 2 usage error. Subcommands add their own codes.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

COMMANDS: dict[str, tuple[str, str]] = {
    "synth": ("hcsim.synth.cli", "generate a synthetic world"),
    "guidance": ("hcsim.guidance.cli", "fit the guidance forecast"),
    "study": ("hcsim.studies.cli", "run a study and write its scorecard"),
    "bench": ("hcsim.bench.cli", "time the synthetic world's presets"),
    "app": ("hcsim.app.cli", "serve the explorer on this machine"),
}

USAGE = "usage: hcsim <command> [options]\n\ncommands:\n" + "\n".join(
    f"  {name:8} {about}" for name, (_, about) in COMMANDS.items()
)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0 if args else 2
    command, rest = args[0], args[1:]
    if command not in COMMANDS:
        print(f"hcsim: unknown command {command!r}\n\n{USAGE}", file=sys.stderr)
        return 2
    module = importlib.import_module(COMMANDS[command][0])
    result: int = module.main(rest)
    return result


if __name__ == "__main__":
    sys.exit(main())
