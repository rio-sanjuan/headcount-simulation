"""Smoke tests for the package layout."""

import importlib

import pytest

import hcsim

LAYERS = [
    "hcsim.model",
    "hcsim.model.core",
    "hcsim.model.ml",
    "hcsim.model.domains",
    "hcsim.model.data",
    "hcsim.model.sql",
    "hcsim.guidance",
    "hcsim.financial",
    "hcsim.engine",
    "hcsim.studies",
    "hcsim.synth",
    "hcsim.bench",
]


@pytest.mark.parametrize("name", LAYERS)
def test_layer_imports(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} needs a docstring stating its dependency rule"


def test_version_comes_from_metadata() -> None:
    assert hcsim.__version__ == importlib.metadata.version("hcsim")
