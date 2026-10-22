"""Layered loading of a configuration's settings.

A setting takes its value from the last layer that gives one:

1. **defaults**, from the configuration's fields;
2. **YAML**: a mapping of field groups, each a mapping of keys, such as
   `promotion: {cycle_months: [3, 9]}`;
3. **the environment**: `HCSIM_CONFIG__<GROUP>__<KEY>`, such as
   `HCSIM_CONFIG__RUN__SIMULATION_MONTHS=24`;
4. **overrides**: `{"<group>.<key>": value}`, from code or a command line.

Text from YAML, the environment, or the command line is read as the setting's type: an integer,
a number, a flag (`true`/`false`, `yes`/`no`, `1`/`0`), an enum value, an ISO date, a
comma-separated list, or `none` for a setting that may be empty. Every problem in every layer
is collected: an unknown group or key, a value that doesn't read as its type, or a required
setting no layer gives. Objects, such as the schema and the models, are passed in code.

`load_settings` returns the merged values with the layer each came from, and `load_config`
builds the configuration from them.

**Records.** `to_record` writes a configuration as plain data: every setting by its path, the
schema's and the guidance's fingerprints, whether it carries award settings, and each model's
artifact version. `from_record` builds it again: the caller passes the schema, the guidance, and
the award settings, which must match what the record names, and each model is re-attached from
its artifact by version. A model built in code has no version, so a configuration holding one
can't be written. Every problem is collected, as in loading.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import os
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

import yaml

from hcsim.engine.config import (
    ARTIFACT_DOMAINS,
    GROUPS,
    MODEL_FIELDS,
    AwardSettings,
    ConfigError,
    DomainModels,
    SimulationConfig,
)
from hcsim.model.core.guidance import GuidanceForecast
from hcsim.model.core.schema import SimulationSchema
from hcsim.model.interface import EntityModel, GroupRateModel
from hcsim.model.ml.generative import EmpiricalAttributeGenerator
from hcsim.model.ml.persistence import ArtifactError, load_model

ENV_PREFIX: Final = "HCSIM_CONFIG__"
RECORD_VERSION: Final = 1
LAYERS: Final = ("default", "yaml", "environment", "override")
_TRUE: Final = frozenset({"true", "yes", "1"})
_FALSE: Final = frozenset({"false", "no", "0"})
_NONE: Final = frozenset({"", "none", "null"})


@dataclass(frozen=True)
class Setting:
    """One layered field of the configuration."""

    name: str
    group: str
    key: str
    hint: Any
    default: Any
    doc: str

    @property
    def path(self) -> str:
        return f"{self.group}.{self.key}"

    @property
    def required(self) -> bool:
        return self.default is dataclasses.MISSING


def settings() -> dict[str, Setting]:
    """Every setting, by its `group.key` path, in field order."""
    hints = typing.get_type_hints(SimulationConfig)
    found = {}
    for f in dataclasses.fields(SimulationConfig):
        group = f.metadata.get("group")
        if group is None:
            continue
        key = f.name if group == "run" else f.name.removeprefix(f"{group}_")
        found[f"{group}.{key}"] = Setting(
            f.name, group, key, hints[f.name], f.default, f.metadata["doc"]
        )
    return found


def _optional(hint: Any) -> tuple[Any, bool]:
    """The type inside `X | None`, and whether None is allowed."""
    if isinstance(hint, types.UnionType) or typing.get_origin(hint) is typing.Union:
        inner = [a for a in typing.get_args(hint) if a is not type(None)]
        return inner[0], len(inner) < len(typing.get_args(hint))
    return hint, False


def coerce(value: Any, hint: Any) -> Any:
    """`value` as `hint`, reading text where it's text. Raises ValueError if it doesn't read."""
    kind, nullable = _optional(hint)
    if nullable and (value is None or (isinstance(value, str) and value.lower() in _NONE)):
        return None
    if typing.get_origin(kind) is tuple:
        (item, _) = typing.get_args(kind)
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",") if p.strip()]
        elif isinstance(value, list | tuple):
            parts = list(value)
        else:
            raise ValueError(f"expected a list, got {value!r}")
        return tuple(coerce(p, item) for p in parts)
    if kind is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in _TRUE | _FALSE:
            return value.lower() in _TRUE
        raise ValueError(f"expected true or false, got {value!r}")
    if kind is int:
        if isinstance(value, bool) or not isinstance(value, int | str):
            raise ValueError(f"expected an integer, got {value!r}")
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"expected an integer, got {value!r}") from None
    if kind is float:
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            raise ValueError(f"expected a number, got {value!r}")
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"expected a number, got {value!r}") from None
    if kind is dt.date:
        if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
            return value
        try:
            return dt.date.fromisoformat(str(value))
        except ValueError:
            raise ValueError(f"expected an ISO date, got {value!r}") from None
    if isinstance(kind, type) and issubclass(kind, Enum):
        try:
            return kind(value)
        except ValueError:
            known = ", ".join(str(m.value) for m in kind)
            raise ValueError(f"expected one of {known}, got {value!r}") from None
    if kind is str:
        if not isinstance(value, str):
            raise ValueError(f"expected text, got {value!r}")
        return value
    raise TypeError(f"no reader for settings of type {kind!r}")  # pragma: no cover


@dataclass(frozen=True)
class Layered:
    """Merged settings: each value by field name, and the layer it came from."""

    values: Mapping[str, Any]
    sources: Mapping[str, str]


def _from_yaml(path: Path, problems: list[str]) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        problems.append(f"yaml: can't read {path}: {error}")
        return {}
    if data is None:
        return {}
    if not isinstance(data, dict):
        problems.append("yaml: the file must be a mapping of field groups")
        return {}
    found = {}
    for group, entries in data.items():
        if group not in GROUPS:
            problems.append(f"yaml: no field group {group!r}; known: {', '.join(GROUPS)}")
        elif not isinstance(entries, dict):
            problems.append(f"yaml: group {group!r} must be a mapping of keys")
        else:
            found.update({f"{group}.{key}": value for key, value in entries.items()})
    return found


def _from_env(env: Mapping[str, str], problems: list[str]) -> dict[str, Any]:
    found = {}
    for name, value in sorted(env.items()):
        if not name.startswith(ENV_PREFIX):
            continue
        group, sep, key = name.removeprefix(ENV_PREFIX).partition("__")
        if not sep or not group or not key:
            problems.append(f"environment: {name} must read {ENV_PREFIX}<GROUP>__<KEY>")
            continue
        found[f"{group.lower()}.{key.lower()}"] = value
    return found


def load_settings(
    *,
    yaml_path: Path | None = None,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Layered:
    """Merge the layers. `env` defaults to the process environment."""
    known = settings()
    problems: list[str] = []
    values = {s.name: s.default for s in known.values() if not s.required}
    sources = dict.fromkeys(values, "default")
    layers = [
        ("yaml", _from_yaml(yaml_path, problems) if yaml_path is not None else {}),
        ("environment", _from_env(os.environ if env is None else env, problems)),
        ("override", dict(overrides or {})),
    ]
    for layer, entries in layers:
        for path, raw in entries.items():
            setting = known.get(path)
            if setting is None:
                group = path.partition(".")[0]
                where = "no field group" if group not in GROUPS else f"group {group!r} has no key"
                problems.append(f"{layer}: {path}: {where}")
                continue
            try:
                values[setting.name] = coerce(raw, setting.hint)
            except ValueError as error:
                problems.append(f"{layer}: {path}: {error}")
                continue
            sources[setting.name] = layer
    for setting in known.values():
        if setting.required and setting.name not in values:
            problems.append(f"{setting.path} is required, and no layer gives it")
    if problems:
        raise ConfigError(problems)
    return Layered(values, sources)


def load_config(
    *,
    schema: SimulationSchema,
    models: DomainModels,
    guidance: GuidanceForecast | None = None,
    awards: AwardSettings | None = None,
    yaml_path: Path | None = None,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> SimulationConfig:
    """The configuration from layered settings and the objects passed in code."""
    layered = load_settings(yaml_path=yaml_path, env=env, overrides=overrides)
    return SimulationConfig(
        schema=schema, models=models, guidance=guidance, awards=awards, **layered.values
    )


#: What each model field must be, to serve its domain.
_SERVES: Final[Mapping[str, type]] = {
    "attrition": EntityModel,
    "external_fill": EntityModel,
    "cancellation": EntityModel,
    "backfill": GroupRateModel,
    "growth": GroupRateModel,
    "attributes": EmpiricalAttributeGenerator,
}


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def to_record(config: SimulationConfig) -> dict[str, Any]:
    """The configuration as plain data, with models by artifact version."""
    unversioned = config.models.unversioned
    if unversioned:
        raise ConfigError(
            [f"models built in code have no version to record: {', '.join(unversioned)}"]
        )
    return {
        "record_version": RECORD_VERSION,
        "settings": {path: _plain(getattr(config, s.name)) for path, s in settings().items()},
        "schema": config.schema.fingerprint(),
        "guidance": None if config.guidance is None else config.guidance.fingerprint(),
        "awards": config.awards is not None,
        "models": dict(config.models.versions),
    }


def _attach(
    versions: Mapping[str, Any], models_root: Path, schema: SimulationSchema, problems: list[str]
) -> DomainModels | None:
    loaded: dict[str, Any] = {}
    for name in MODEL_FIELDS:
        version = versions.get(name)
        if not isinstance(version, str):
            problems.append(f"the record names no version for the {name} model")
            continue
        domain = ARTIFACT_DOMAINS.get(name, name)
        try:
            model, artifact = load_model(models_root / domain / version)
        except (ArtifactError, OSError) as error:
            problems.append(f"{name}: {error}")
            continue
        recorded = artifact.config.get("schema")
        if recorded is not None and recorded != schema.fingerprint():
            problems.append(f"{name} {version} was trained on schema {recorded}")
        if not isinstance(model, _SERVES[name]):
            problems.append(
                f"{name} {version} holds a {type(model).__name__}, which can't serve it"
            )
        loaded[name] = model
    if len(loaded) < len(MODEL_FIELDS):
        return None
    return DomainModels(**loaded, versions={n: str(versions[n]) for n in MODEL_FIELDS})


def from_record(
    record: Mapping[str, Any],
    *,
    schema: SimulationSchema,
    models_root: Path,
    guidance: GuidanceForecast | None = None,
    awards: AwardSettings | None = None,
) -> SimulationConfig:
    """Build a configuration from its record, re-attaching each model by version."""
    problems: list[str] = []
    if record.get("record_version") != RECORD_VERSION:
        raise ConfigError([f"unsupported configuration record {record.get('record_version')!r}"])
    if record.get("schema") != schema.fingerprint():
        problems.append(
            f"the record was written for schema {record.get('schema')}, not {schema.fingerprint()}"
        )
    wanted = record.get("guidance")
    given = None if guidance is None else guidance.fingerprint()
    if wanted != given:
        problems.append(f"the record names guidance {wanted}, but the call passes {given}")
    if bool(record.get("awards")) != (awards is not None):
        need = "needs" if record.get("awards") else "has no"
        problems.append(f"the record {need} award settings, which the call doesn't match")
    try:
        layered = load_settings(env={}, overrides=record.get("settings", {}))
    except ConfigError as error:
        problems += [f"settings: {p}" for p in error.problems]
        layered = None
    models = _attach(record.get("models", {}), models_root, schema, problems)
    if problems or layered is None or models is None:
        raise ConfigError(problems)
    return SimulationConfig(
        schema=schema, models=models, guidance=guidance, awards=awards, **layered.values
    )
