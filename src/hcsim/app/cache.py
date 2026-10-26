"""The explorer's kept answers: each preset's answer on disk, so a later start answers at once.

`hcsim app prepare` warms them ahead, and `hcsim app serve` reads the same ones, so a first
reader never waits. An answer is kept under a key of everything that made it: the world's
contents (a hash of its file, not its name), the engine's version, the demo, and its parameters.
A changed world or a new version misses the cache, and runs again.

**Where they're kept:** in `--cache DIR` when it's given, or in `~/.cache/hcsim/explorer`
(under `$XDG_CACHE_HOME` when that's set). A development version (one with `.dev` in it)
doesn't change with every commit, so it keeps answers only in a directory given by name: a code
change then never serves a stale answer.

Only the presets are kept on disk. Adjust's answers live in the server's memory alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

#: How much of the world's file is read at a time, to hash it.
CHUNK = 1 << 20


def default_dir() -> Path:
    """Where answers are kept unless a directory is given."""
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "hcsim" / "explorer"


def is_development(version: str) -> bool:
    """Whether `version` is a development version, which doesn't change with every commit."""
    return ".dev" in version


def file_hash(path: Path) -> str:
    """The SHA-256 of a file's contents."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


class Kept:
    """The answers kept for one world and version, in `root`, or nowhere when `root` is None."""

    def __init__(self, root: Path | None, world: Path, version: str) -> None:
        self.root = root
        self.version = version
        # Hashing a large world takes a few seconds, so it's done only where answers are kept.
        self.world = file_hash(world) if root is not None else ""

    @classmethod
    def where(cls, asked: Path | None, world: Path, version: str) -> Kept:
        """The kept answers for `world`: in `asked` if it's given, otherwise in the default
        directory, unless `version` is a development one, which keeps nothing on disk."""
        if asked is not None:
            return cls(asked, world, version)
        return cls(None if is_development(version) else default_dir(), world, version)

    def _path(self, demo: str, params: dict[str, Any]) -> Path | None:
        if self.root is None:
            return None
        # Every parameter as a float, so 1 and 1.0 are one run.
        key = {
            "world": self.world,
            "version": self.version,
            "demo": demo,
            "params": {k: float(v) for k, v in sorted(params.items())},
        }
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        return self.root / demo / f"{digest}.json"

    def get(self, demo: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """The answer kept for `demo` with `params`, or None. A file that can't be read is a
        miss, so the answer runs again and replaces it."""
        path = self._path(demo, params)
        if path is None:
            return None
        try:
            answer: dict[str, Any] = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        return answer

    def put(self, demo: str, params: dict[str, Any], answer: dict[str, Any]) -> None:
        """Keeps `answer` for `demo` with `params`. It's written beside its place and moved into
        it, so a reader never sees half of one."""
        path = self._path(demo, params)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.partial")
        partial.write_text(json.dumps(answer))
        os.replace(partial, path)
