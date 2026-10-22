"""Result digests: one hash over every value a trajectory's result holds, bit for bit.

Two results have the same digest when every array has the same dtype, shape, and bytes, every
frame the same columns, dtypes, index, and values, and every other field the same value. A
streamed run records each trajectory's digest as it hands the trajectory out, so its parity
with a standard run can be checked without holding either run's results.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def digest(value: object) -> str:
    """The SHA-256 hex digest of `value`'s contents."""
    h = hashlib.sha256()
    _feed(h, value)
    return h.hexdigest()


def _feed(h: Any, value: object) -> None:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        h.update(f"array:{array.dtype.str}:{array.shape}:".encode())
        if array.dtype == object:
            h.update(repr(array.ravel().tolist()).encode())
        else:
            h.update(array.tobytes())
    elif isinstance(value, pd.DataFrame):
        h.update(f"frame:{len(value)}:".encode())
        _feed(h, value.index.to_numpy())
        for name in value.columns:
            column = value[name]
            h.update(f"column:{name}:{column.dtype}:".encode())
            if isinstance(column.dtype, pd.CategoricalDtype):
                _feed(h, column.cat.categories.to_numpy())
                _feed(h, column.cat.codes.to_numpy())
            else:
                _feed(h, column.to_numpy())
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        h.update(f"{type(value).__name__}:".encode())
        for f in dataclasses.fields(value):
            h.update(f"{f.name}=".encode())
            _feed(h, getattr(value, f.name))
    elif isinstance(value, Mapping):
        h.update(f"mapping:{len(value)}:".encode())
        for key in sorted(value, key=repr):
            h.update(f"{key!r}=".encode())
            _feed(h, value[key])
    elif isinstance(value, tuple | list):
        h.update(f"sequence:{len(value)}:".encode())
        for item in value:
            _feed(h, item)
    else:
        h.update(f"{type(value).__name__}:{value!r};".encode())
