"""The transaction vocabulary: the codes on the workforce facts, and what each one means.

`TransType` names the codes the data contract defines. What a code *does* is data, not code:
its disposition (measured, excluded, or ignored) and its signed headcount amount come from the
warehouse's `dim_trans_type`, read into a `TransTypeCatalog`. Readers ask the catalog how to
treat a code, so a contract change to a code's disposition needs no change here.

A code the facts carry but the catalog doesn't define has no disposition. Counting it, or
skipping it, would be a guess, so the catalog refuses it by name.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class TransType(StrEnum):
    """The transaction codes of data contract 1.0."""

    ACTIVE = "ACTIVE"
    HIRE = "HIRE"
    TERM = "TERM"
    RIF = "RIF"
    TERM_PENDING = "TERM_PENDING"
    XFER_OUT = "XFER_OUT"
    XFER_IN = "XFER_IN"
    REORG_OUT = "REORG_OUT"
    REORG_IN = "REORG_IN"
    PROMO = "PROMO"
    RECLASS = "RECLASS"
    CONV_OUT = "CONV_OUT"
    CONV_IN = "CONV_IN"
    CTG_START = "CTG_START"
    CTG_END = "CTG_END"
    ADMIN = "ADMIN"
    HIRE_REV = "HIRE_REV"
    TERM_REV = "TERM_REV"
    HIRE_PENDING = "HIRE_PENDING"
    SYS_ADJ = "SYS_ADJ"


class Disposition(StrEnum):
    """How readers treat a code: measured codes count, excluded codes belong to the contingent
    pool that headcount measures leave out, and ignored codes have no effect."""

    MEASURED = "measured"
    EXCLUDED = "excluded"
    IGNORED = "ignored"


class TransTypeError(ValueError):
    """A catalog that can't be trusted, or a code it doesn't define."""


@dataclass(frozen=True)
class TransTypeInfo:
    code: str
    disposition: Disposition
    amount: int
    description: str = ""


class TransTypeCatalog:
    """Every code's disposition and amount, as the warehouse defines them."""

    def __init__(self, entries: Iterable[TransTypeInfo]) -> None:
        listed = list(entries)
        problems = []
        seen: dict[str, TransTypeInfo] = {}
        for entry in listed:
            if entry.code in seen:
                problems.append(f"{entry.code!r} is defined twice")
            if entry.amount not in (-1, 0, 1):
                problems.append(f"{entry.code!r} has amount {entry.amount}, not -1, 0, or 1")
            seen[entry.code] = entry
        if problems:
            raise TransTypeError("bad transaction catalog: " + "; ".join(problems))
        self._entries: Mapping[str, TransTypeInfo] = MappingProxyType(seen)

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]]) -> TransTypeCatalog:
        """From `dim_trans_type` rows: `trans_type`, `disposition`, `amount`, `description`.
        Every unknown disposition is reported at once."""
        entries, problems = [], []
        for row in rows:
            code = str(row["trans_type"])
            try:
                disposition = Disposition(str(row["disposition"]))
            except ValueError:
                known = ", ".join(d.value for d in Disposition)
                problems.append(f"{code!r} has disposition {row['disposition']!r}; known: {known}")
                continue
            amount = int(row["amount"])
            entries.append(
                TransTypeInfo(code, disposition, amount, str(row.get("description", "")))
            )
        if problems:
            raise TransTypeError("bad transaction catalog: " + "; ".join(problems))
        return cls(entries)

    @classmethod
    def from_warehouse(cls, con: Any) -> TransTypeCatalog:
        """Read `dim_trans_type` through a DuckDB connection."""
        frame = con.execute("SELECT * FROM dim_trans_type").df()
        return cls.from_rows(frame.to_dict("records"))

    def __contains__(self, code: object) -> bool:
        return str(code) in self._entries

    def __iter__(self) -> Iterator[TransTypeInfo]:
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def info(self, code: str) -> TransTypeInfo:
        try:
            return self._entries[str(code)]
        except KeyError:
            raise TransTypeError(f"transaction code {code!r} isn't in the catalog") from None

    def disposition(self, code: str) -> Disposition:
        return self.info(code).disposition

    def amount(self, code: str) -> int:
        return self.info(code).amount

    def codes(self, disposition: Disposition) -> frozenset[str]:
        return frozenset(e.code for e in self._entries.values() if e.disposition == disposition)

    def check(self, codes: Iterable[object]) -> None:
        """Refuse codes the catalog doesn't define, naming every one."""
        unknown = sorted({str(c) for c in codes} - set(self._entries))
        if unknown:
            raise TransTypeError(f"transaction codes missing from the catalog: {unknown}")

    def known_types(self) -> frozenset[TransType]:
        """The contract's codes that this catalog defines."""
        return frozenset(t for t in TransType if t.value in self._entries)
