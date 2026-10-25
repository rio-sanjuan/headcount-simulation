"""The pipeline framework: ordered steps over a shared context.

A pipeline is an ordered tuple of steps, and the context is what they share: named slots of
data, a report section per step, and a record of each step's run.

Each step declares the slots it reads (`needs`) and the slots it writes (`gives`). A step that
both needs and gives a slot rewrites it in place, as reconciliation rewrites the forecasts;
otherwise each slot has one writer. Building a pipeline checks every declaration it can without
running: step names are unique, no two steps write a fresh slot of the same name, and before a
run, every slot a step reads is seeded into the context or given by an earlier step.

Running it, each step **validates before it executes**. It checks the slots it's about to read
and reports every problem at once, and a step with any problem stops the run before it does any
work, with an error that names the step and each problem. A step then executes, returning
exactly the slots it declared, and the framework writes them into the context. A step that
returns one it didn't declare, or leaves one out, is a defect, and the run stops there too.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class PipelineError(ValueError):
    """A pipeline that can't run, or a step that can't: every problem, with the step named."""

    def __init__(self, problems: Sequence[str], step: str | None = None) -> None:
        self.problems = list(problems)
        self.step = step
        where = "pipeline" if step is None else f"step {step!r}"
        super().__init__(f"{where} can't run:\n  - " + "\n  - ".join(self.problems))


@dataclass(frozen=True)
class StepRecord:
    """One step's run: its name, the slots it wrote, and its seconds."""

    step: str
    gave: tuple[str, ...]
    seconds: float


@dataclass
class Context:
    """The state a pipeline's steps share."""

    slots: dict[str, Any] = field(default_factory=dict)
    #: Each step's report section, by step name.
    report: dict[str, dict[str, Any]] = field(default_factory=dict)
    records: list[StepRecord] = field(default_factory=list)

    def __getitem__(self, slot: str) -> Any:
        try:
            return self.slots[slot]
        except KeyError:
            raise KeyError(
                f"no slot {slot!r} in the context; it holds {sorted(self.slots)}"
            ) from None

    def __contains__(self, slot: object) -> bool:
        return slot in self.slots

    def note(self, step: str, **entries: Any) -> None:
        """Add entries to `step`'s report section."""
        self.report.setdefault(step, {}).update(entries)

    def fork(self, slots: Mapping[str, Any] | None = None) -> Context:
        """A copy, with `slots` added, for more steps to run on while this one stays as it was.
        The slots' values are shared, not copied: a step returns new values rather than
        changing the ones it reads."""
        return Context(
            {**self.slots, **(slots or {})},
            {step: dict(section) for step, section in self.report.items()},
            list(self.records),
        )


class Step(Protocol):
    """One stage of a pipeline."""

    @property
    def name(self) -> str: ...

    @property
    def needs(self) -> tuple[str, ...]: ...

    @property
    def gives(self) -> tuple[str, ...]: ...

    def validate(self, ctx: Context) -> list[str]:
        """Every problem with the context this step is about to read; none means it can run."""
        ...

    def execute(self, ctx: Context) -> Mapping[str, Any]:
        """Do the step's work, returning exactly the slots it gives."""
        ...


@dataclass(frozen=True)
class Pipeline:
    steps: tuple[Step, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        problems = []
        names = [s.name for s in self.steps]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            problems.append(f"step names must be unique: {repeated} repeat")
        writers: dict[str, str] = {}
        for step in self.steps:
            for slot in step.gives:
                if slot in step.needs:
                    continue  # a rewrite
                if slot in writers:
                    problems.append(
                        f"{step.name!r} gives {slot!r}, which {writers[slot]!r} already gives"
                    )
                writers.setdefault(slot, step.name)
        if problems:
            raise PipelineError(problems)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.steps)

    def check(self, seeded: Iterable[str]) -> list[str]:
        """Every slot a step reads that neither the seed nor an earlier step gives."""
        available = set(seeded)
        problems = []
        for step in self.steps:
            missing = [slot for slot in step.needs if slot not in available]
            if missing:
                problems.append(f"{step.name!r} needs {missing}, which nothing before it gives")
            available.update(step.gives)
        return problems

    def run(self, ctx: Context) -> Context:
        """Run every step in order over `ctx`, which they update in place."""
        problems = self.check(ctx.slots)
        if problems:
            raise PipelineError(problems)
        for step in self.steps:
            problems = step.validate(ctx)
            if problems:
                raise PipelineError(problems, step.name)
            start = time.perf_counter()
            products = dict(step.execute(ctx))
            extra = sorted(set(products) - set(step.gives))
            missing = sorted(set(step.gives) - set(products))
            if extra or missing:
                found = []
                if extra:
                    found.append(f"returned undeclared slots {extra}")
                if missing:
                    found.append(f"didn't return {missing}")
                raise PipelineError(found, step.name)
            ctx.slots.update(products)
            ctx.records.append(StepRecord(step.name, step.gives, time.perf_counter() - start))
        return ctx
