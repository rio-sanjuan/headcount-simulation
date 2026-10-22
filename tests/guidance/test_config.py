"""Tests for the guidance configuration and the registries it's validated against."""

from __future__ import annotations

from pathlib import Path

import pytest

from hcsim.cli import main as hcsim
from hcsim.guidance.config import GuidanceConfig, GuidanceConfigError
from hcsim.guidance.models import MODELS, SeasonalNaive
from hcsim.guidance.reconcile import RECONCILERS, reconcile_greedy
from hcsim.guidance.registry import Entry, Registry, RegistryError
from hcsim.guidance.selection import COMBINERS, inverse_error_weights

# --- registries ---------------------------------------------------------------------------------


def toy() -> Registry[str]:
    return Registry(
        "widget",
        (
            Entry("plain", lambda: "built"),
            Entry("extra.thing", lambda: "never", extra="fancy", module="hcsim_no_such_module"),
            Entry("present", lambda: "here", extra="core", module="numpy"),
        ),
    )


def test_a_registry_builds_what_it_names() -> None:
    registry = toy()
    assert registry.names == ("extra.thing", "plain", "present")
    assert registry.build("plain") == "built"
    assert registry.build("present") == "here"  # its module is installed
    assert "plain" in registry
    assert "other" not in registry


def test_a_registry_names_every_problem_and_how_to_fix_it() -> None:
    assert toy().problems(["plain", "nope", "extra.thing"]) == [
        "unknown widget 'nope'; known: extra.thing, plain, present",
        "widget 'extra.thing' needs the 'fancy' extra; install it with `uv sync --extra fancy`",
    ]
    with pytest.raises(RegistryError, match="needs the 'fancy' extra"):
        toy().build("extra.thing")


def test_a_registry_refuses_a_name_twice() -> None:
    registry = toy()
    with pytest.raises(RegistryError, match="already has 'plain'"):
        registry.register(Entry("plain", lambda: "again"))


def test_a_namespaced_registry_names_its_entries_by_namespace() -> None:
    registry: Registry[str] = Registry(
        "widget",
        (Entry(f"{ns}.{name}", lambda: "w") for ns in ("alpha", "beta") for name in "ABCDEFG"),
        namespaced=True,
    )
    assert registry.namespaces == ("alpha", "beta")
    assert registry.in_namespace("beta") == tuple(f"beta.{n}" for n in "ABCDEFG")
    with pytest.raises(RegistryError, match=r"<namespace>\.<Name>, not 'plain'"):
        registry.register(Entry("plain", lambda: "w"))
    # A large registry names the namespace's entries, or every namespace, not every entry.
    wrong_case, missing, elsewhere = registry.problems(["alpha.a", "beta.Z", "gamma.A"])
    assert wrong_case.startswith("unknown widget 'alpha.a'; did you mean 'alpha.A'")
    assert wrong_case.endswith("; alpha has " + ", ".join(f"alpha.{n}" for n in "ABCDEFG"))
    assert missing.startswith("unknown widget 'beta.Z'")
    assert missing.endswith("; beta has " + ", ".join(f"beta.{n}" for n in "ABCDEFG"))
    assert elsewhere == "unknown widget 'gamma.A'; known namespaces: alpha, beta"


def test_the_model_registry_is_namespaced() -> None:
    assert MODELS.namespaced
    assert set(MODELS.namespaces) == {"native", "statsforecast", "darts", "sktime", "prophet"}
    assert MODELS.problems(["seasonal_naive"]) == [
        "unknown model 'seasonal_naive'; did you mean 'native.SeasonalNaive'; known namespaces: "
        + ", ".join(MODELS.namespaces)
    ]
    assert MODELS.problems(["statsforecast.autoarima"])[0].startswith(
        "unknown model 'statsforecast.autoarima'; did you mean 'statsforecast.AutoARIMA'"
    )


def test_the_pipeline_s_registries() -> None:
    assert {
        "native.Drift",
        "native.HistoricalAverage",
        "native.LastObserved",
        "native.SeasonalNaive",
    } <= set(MODELS.names)
    assert isinstance(MODELS.build("native.SeasonalNaive"), SeasonalNaive)
    assert COMBINERS.build("inverse_error") is inverse_error_weights
    assert RECONCILERS.build("greedy") is reconcile_greedy


# --- the configuration --------------------------------------------------------------------------


def test_the_defaults_load_and_map_onto_the_steps_settings() -> None:
    config = GuidanceConfig()
    assert config.problems() == []
    assert config.reconcile
    assert config.tolerance == 1.0
    selection = config.selection
    assert selection.suite == ("native.SeasonalNaive", "native.HistoricalAverage")
    assert (selection.top_k, selection.combination) == (2, "inverse_error")
    assert selection.evaluation.min_length == 42


def test_a_configuration_reads_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "guidance.yaml"
    path.write_text(
        "horizon: 6\ngrouping: TOPLINE\nmodels: [native.SeasonalNaive]\n"
        "combination: {top_k: 1}\nevaluation: {min_train_months: 24}\n"
        "reconciliation: {method: null}\n"
    )
    config = GuidanceConfig.load(path)
    assert (config.horizon, config.grouping, config.models) == (
        6,
        "TOPLINE",
        ("native.SeasonalNaive",),
    )
    assert not config.reconcile
    assert config.selection.evaluation.min_length == 30
    assert GuidanceConfig.from_mapping(config.model_dump()) == config


def test_an_empty_file_is_the_defaults(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert GuidanceConfig.load(path) == GuidanceConfig()


def test_every_unregistered_name_is_refused_at_load_at_once() -> None:
    with pytest.raises(GuidanceConfigError) as error:
        GuidanceConfig.from_mapping(
            {
                "models": ["native.SeasonalNaive", "prophet", "native.SeasonalNaive"],
                "combination": {"method": "median"},
                "reconciliation": {"method": "mint"},
                "grouping": "cost_type",
            }
        )
    lines = str(error.value).splitlines()
    assert lines[0] == "invalid guidance configuration:"
    assert lines[1:] == [
        "  - models: unknown model 'prophet'; did you mean 'prophet.Prophet'; prophet has "
        "prophet.Prophet",
        "  - models: each model is named once; ['native.SeasonalNaive'] repeat",
        "  - combination.method: unknown combination method 'median'; known: inverse_error",
        "  - reconciliation.method: unknown reconciliation method 'mint'; known: greedy",
        "  - grouping: 'cost_type' can't group by; it permits filter, population",
    ]


def test_fields_are_typed_ranged_and_spelled_right() -> None:
    with pytest.raises(GuidanceConfigError) as error:
        GuidanceConfig.from_mapping(
            {"horizon": 0, "models": [], "evaluation": {"n_splits": 0}, "horizn": 3}
        )
    message = str(error.value)
    for field in ("horizon", "models", "evaluation.n_splits", "horizn"):
        assert f"  - {field}: " in message


def test_an_unreadable_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(GuidanceConfigError, match="can't read the configuration"):
        GuidanceConfig.load(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("horizon: [")
    with pytest.raises(GuidanceConfigError, match="can't read the configuration"):
        GuidanceConfig.load(bad)


def test_the_command_refuses_a_configuration_before_reading_the_world(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = tmp_path / "world.duckdb"
    world.write_bytes(b"")  # never opened: the configuration fails first
    config = tmp_path / "guidance.yaml"
    config.write_text("models: [prophet]\n")
    args = ["guidance", "fit", "--world", str(world), "--out", str(tmp_path / "out")]
    with pytest.raises(SystemExit) as stop:
        hcsim([*args, "--config", str(config)])
    assert stop.value.code == 2
    assert "unknown model 'prophet'" in capsys.readouterr().err
