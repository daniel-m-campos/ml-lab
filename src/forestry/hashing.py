"""Content hashing of declarations: canonical JSON over values and registered step references.

A declaration may hold values, dataclasses, and references to registered steps or scorers. A
function without a registration is refused, so closures and lambdas never reach a hash.

Examples
--------
>>> content_hash({"b": 1, "a": [1, 2]}) == content_hash({"a": [1, 2], "b": 1})
True
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import inspect
import json
import pathlib
from collections.abc import Callable
from typing import Any

import numpy as np

STEP_ATTR = "__forestry_step__"
HASH_LEN = 16


@dataclasses.dataclass(frozen=True)
class StepRef:
    """Where a registered function lives and the sha of the file that defines it."""

    module: str
    qualname: str
    file_sha: str

    @property
    def path(self) -> str:
        return f"{self.module}:{self.qualname}"


# Public Functions =================================================================================


def register(func: Callable, kind: str, **meta: Any) -> Callable:
    """Mark a function as a hashable step or scorer and return it unchanged."""
    source = pathlib.Path(inspect.getsourcefile(func) or "")
    file_sha = (
        hashlib.sha256(source.read_bytes()).hexdigest()[:HASH_LEN] if source.is_file() else "nofile"
    )
    setattr(func, STEP_ATTR, StepRef(func.__module__, func.__qualname__, file_sha))
    func.__forestry_kind__ = kind  # type: ignore[attr-defined]
    func.__forestry_meta__ = meta  # type: ignore[attr-defined]
    return func


def step_ref(func: Callable) -> StepRef:
    """The registration of a step, or a TypeError for an unregistered callable."""
    ref = getattr(func, STEP_ATTR, None)
    if ref is None:
        raise TypeError(f"{func!r} is not a registered step; closures and lambdas cannot be hashed")
    return ref


def canonical(obj: Any) -> Any:
    """Reduce a declaration to JSON-serializable values with a stable form."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    if callable(obj):
        return {"__step__": step_ref(obj).path, "sha": step_ref(obj).file_sha}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = {
            f.name: canonical(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
            if not f.metadata.get("label")
        }
        return {"__type__": type(obj).__qualname__, **fields}
    if isinstance(obj, dict):
        return {str(k): canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [canonical(v) for v in obj]
    raise TypeError(f"cannot serialize {type(obj).__name__} into a declaration")


def content_hash(obj: Any) -> str:
    """Hex digest of the canonical form, 16 characters."""
    payload = json.dumps(canonical(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:HASH_LEN]


def bytes_hash(payload: bytes) -> str:
    """Full sha256 hex of raw bytes, the blob store key."""
    return hashlib.sha256(payload).hexdigest()
