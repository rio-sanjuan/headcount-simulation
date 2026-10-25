"""Check that every locked npm package's license is compatible with Apache-2.0.

Reads the explorer's `package-lock.json`, whose every package declares its license, so nothing
has to be fetched or installed. The rule is the Python check's (`licenses.py`): a package fails
when its license is strong copyleft (GPL or AGPL, SSPL, the Business Source License, or the
Commons Clause); one compatible `OR` alternative is enough. A package that declares no license
is listed for review but doesn't fail.

Exit codes: 0 compatible, 1 incompatible licenses found, 2 the lockfile couldn't be read.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tools.licenses import COPYLEFT


def verdict(license_: str) -> str:
    """`ok`, `incompatible`, or `review` for one SPDX expression."""
    if not license_.strip():
        return "review"
    alternatives = [a.strip(" ()") for a in re.split(r"\s+OR\s+", license_)]
    if any(not COPYLEFT.search(a) for a in alternatives):
        return "ok"
    return "incompatible"


def check(lock: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    """Each locked package, by name and version, with its license and verdict."""
    packages = lock.get("packages")
    if not isinstance(packages, dict):
        raise ValueError("a lockfile version 2 or later lists its packages")
    out = {}
    for path, meta in packages.items():
        if not path or not isinstance(meta, dict):
            continue  # the root project itself
        name = path.rsplit("node_modules/", 1)[-1]
        license_ = meta.get("license", "")
        text = license_ if isinstance(license_, str) else json.dumps(license_)
        out[f"{name}@{meta.get('version', '?')}"] = (text, verdict(text))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lockfile", type=Path)
    args = parser.parse_args(argv)
    try:
        found = check(json.loads(args.lockfile.read_text()))
    except (OSError, ValueError) as error:
        print(f"npm licenses: can't read {args.lockfile}: {error}")
        return 2
    for package, (license_, result) in sorted(found.items()):
        if result != "ok":
            print(f"{result:12} {package}: {license_ or '-'}")
    bad = sum(result == "incompatible" for _, result in found.values())
    print(f"npm licenses: {len(found)} packages, {bad} incompatible with Apache-2.0")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
