"""Check that every locked dependency's license is compatible with Apache-2.0.

Reads `uv export` output (every extra and group) and fetches each pinned version's metadata
from PyPI, so nothing has to be installed. Extras such as PyTorch are covered too.

A package fails when its license signals are all strong copyleft: GPL or AGPL, SSPL, the Business
Source License, or the Commons Clause. LGPL and MPL are compatible as dependencies. When a
package declares an SPDX expression, one compatible `OR` alternative is enough. Packages with no
license signal, or with mixed signals, are listed for review but don't fail.

Exit codes: 0 compatible, 1 incompatible licenses found, 2 metadata couldn't be fetched.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)")
COPYLEFT = re.compile(
    r"\bA?GPL|GNU (Affero )?General Public|\bSSPL\b|Server Side Public|"
    r"Commons Clause|\bBUSL\b|Business Source",
    re.IGNORECASE,
)
# Longer `license` fields usually hold the whole license text, which can mention the GPL in
# passing, so only short ones count as a signal.
MAX_LICENSE_FIELD = 200


class Verdict(Enum):
    OK = "ok"
    INCOMPATIBLE = "incompatible"
    REVIEW = "review"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Metadata:
    name: str
    version: str
    expression: str
    license: str
    classifiers: tuple[str, ...]

    def signals(self) -> list[str]:
        if self.expression:
            return [self.expression]
        found = [c.removeprefix("License :: ") for c in self.classifiers if c.startswith("License")]
        if self.license and len(self.license) <= MAX_LICENSE_FIELD:
            found.append(self.license)
        return found


def classify(meta: Metadata) -> Verdict:
    if meta.expression:
        alternatives = re.split(r"\s+OR\s+", meta.expression.strip("() "), flags=re.IGNORECASE)
        return (
            Verdict.OK
            if any(not COPYLEFT.search(a) for a in alternatives)
            else (Verdict.INCOMPATIBLE)
        )
    signals = [s for s in meta.signals() if s.strip() and s.upper() != "UNKNOWN"]
    if not signals:
        return Verdict.UNKNOWN
    copyleft = [bool(COPYLEFT.search(s)) for s in signals]
    if all(copyleft):
        return Verdict.INCOMPATIBLE
    return Verdict.REVIEW if any(copyleft) else Verdict.OK


def parse_pins(requirements: str) -> list[tuple[str, str]]:
    pins = {
        (m.group(1), m.group(2))
        for line in requirements.splitlines()
        if (m := PIN.match(line.strip()))
    }
    return sorted(pins)


def fetch(name: str, version: str, attempts: int = 3) -> Metadata:
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                info = json.load(response)["info"]
            return Metadata(
                name,
                version,
                info.get("license_expression") or "",
                info.get("license") or "",
                tuple(info.get("classifiers") or ()),
            )
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("requirements", type=Path, help="output of `uv export`")
    args = parser.parse_args(argv)
    pins = parse_pins(args.requirements.read_text())
    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            metas = list(pool.map(lambda pin: fetch(*pin), pins))
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"licenses: couldn't fetch metadata from PyPI: {error}", file=sys.stderr)
        return 2
    verdicts = {meta: classify(meta) for meta in metas}
    for verdict in (Verdict.INCOMPATIBLE, Verdict.REVIEW, Verdict.UNKNOWN):
        for meta, v in verdicts.items():
            if v is verdict:
                print(f"{verdict.value:12} {meta.name}=={meta.version}: {meta.signals() or '-'}")
    bad = sum(v is Verdict.INCOMPATIBLE for v in verdicts.values())
    print(f"licenses: {len(metas)} packages, {bad} incompatible with Apache-2.0")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
