"""Determinism: one configuration and seed give one world, in any process."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest

from hcsim.synth.config import QuirkTruth, preset
from hcsim.synth.world import World, digest, generate_world, write_world

EVERY_QUIRK = QuirkTruth(
    backdated_share=0.2,
    correction_share=0.2,
    no_show_pair_share=1.0,
    pending_hire_rows=True,
    aggregate_only_share=0.2,
    snapshot_gap_share=0.2,
    state_seam_month="2023-07",
    cost_center_aliases=True,
)


@pytest.fixture(scope="module")
def world() -> World:
    return generate_world(preset("tiny", seed=7))


def test_a_seed_gives_one_world_and_another_seed_another() -> None:
    config = dataclasses.replace(preset("tiny", seed=7), quirks=EVERY_QUIRK)
    first = digest(generate_world(config))
    generate_world(preset("tiny", seed=8))  # another world in between leaves no trace
    assert digest(generate_world(config)) == first
    assert digest(generate_world(dataclasses.replace(config, seed=8))) != first


def test_the_digest_sees_every_table_and_the_manifest(world: World) -> None:
    base = digest(world)

    def changed(**parts: Any) -> str:
        return digest(dataclasses.replace(world, **parts))

    for name in ("dim_org", "workforce_txn", "equity_expense"):
        frame = world.tables[name]
        assert changed(tables=world.tables | {name: frame.iloc[:-1]}) != base
        reordered = frame[list(reversed(frame.columns))]
        assert changed(tables=world.tables | {name: reordered}) != base
    plan = world.truth_tables["latent_plan"]
    assert changed(truth_tables=world.truth_tables | {"latent_plan": plan.iloc[1:]}) != base
    assert changed(truth=world.truth | {"generator_version": -1}) != base
    assert changed() == base


@pytest.mark.slow
def test_the_world_is_the_same_under_any_hash_seed(world: World) -> None:
    """Nothing depends on the iteration order of sets or dicts keyed by strings, which
    Python randomizes per process."""
    code = (
        "from hcsim.synth.config import preset\n"
        "from hcsim.synth.world import digest, generate_world\n"
        "print(digest(generate_world(preset('tiny', seed=7))))\n"
    )
    digests = {
        subprocess.run(
            [sys.executable, "-c", code],
            env=os.environ | {"PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for seed in ("0", "4242")
    }
    assert digests == {digest(world)}


def test_each_write_holds_the_same_rows(world: World, tmp_path: Path) -> None:
    frames = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.duckdb"
        write_world(world, path)
        with duckdb.connect(str(path), read_only=True) as con:
            listed = con.execute(
                "SELECT table_schema, table_name FROM information_schema.tables ORDER BY ALL"
            ).fetchall()
            frames.append(
                {
                    f"{schema}.{table}": con.execute(f'SELECT * FROM {schema}."{table}"').df()
                    for schema, table in listed
                }
            )
    a, b = frames
    assert list(a) == list(b)
    assert len(a) == len(world.tables) + len(world.truth_tables) + 2  # the view and manifest
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name], obj=name)
