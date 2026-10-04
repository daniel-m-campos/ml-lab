"""Identities: canonical serialization of declarations, content hashes, ULIDs, code identity.

A declaration is hashed by its canonical form: dataclass fields minus labels, registered steps by
dotted path. Code identity is separate: ``import_shas`` gives the git blob sha of every repo
module a set of steps imports, computed the way ``git hash-object`` does, so dirty files count.

Examples
--------
>>> content_hash({"a": 1}) == content_hash({"a": 1})
True
>>> len(ulid())
26
"""

from __future__ import annotations

import dataclasses
import datetime
import functools
import hashlib
import importlib.metadata
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import types
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

STEP_ATTR = "__forestry_step__"
HASH_LEN = 16
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


@dataclasses.dataclass(frozen=True)
class StepRef:
    """Where a registered function lives: ``module:qualname``."""

    module: str
    qualname: str

    @property
    def path(self) -> str:
        return f"{self.module}:{self.qualname}"


# Declarations =====================================================================================


def register(func: Callable, kind: str, **meta: Any) -> Callable:
    """Mark a function as a hashable step or scorer and return it unchanged."""
    setattr(func, STEP_ATTR, StepRef(func.__module__, func.__qualname__))
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
        return {"__step__": step_ref(obj).path}
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


# Ids ==============================================================================================

_last_ulid: list[int] = [0, 0]


def ulid() -> str:
    """A 26-character ULID, monotonic within the process."""
    ms = int(time.time() * 1000)
    if ms <= _last_ulid[0]:
        ms = _last_ulid[0]
        rand = _last_ulid[1] + 1
    else:
        rand = int.from_bytes(os.urandom(10), "big")
    _last_ulid[0], _last_ulid[1] = ms, rand
    value = (ms << 80) | rand
    return "".join(CROCKFORD[(value >> (5 * i)) & 31] for i in reversed(range(26)))


# Code identity ====================================================================================


def git_blob_sha(data: bytes) -> str:
    """The sha git would give these bytes as a blob."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def repo_root(start: pathlib.Path) -> pathlib.Path:
    """The git toplevel containing ``start``, or ``start``'s directory outside a repo."""
    directory = start if start.is_dir() else start.parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=True,
        )
        return pathlib.Path(out.stdout.strip()).resolve()
    except (OSError, subprocess.CalledProcessError):
        return directory.resolve()


def closure(funcs: Iterable[Callable], code_root: pathlib.Path) -> list[types.ModuleType]:
    """Every module reachable from the steps' modules through module-level names.

    Modules outside ``code_root`` are reached but not expanded, so a third-party package appears
    once and its internals are never walked.
    """
    root = code_root.resolve()
    queue = [sys.modules[f.__module__] for f in funcs if f.__module__ in sys.modules]
    seen: dict[str, types.ModuleType] = {}
    while queue:
        module = queue.pop()
        if module.__name__ in seen:
            continue
        seen[module.__name__] = module
        if _module_path(module, root) is None:
            continue
        for value in vars(module).values():
            if isinstance(value, types.ModuleType):
                queue.append(value)
            elif isinstance(getattr(value, "__module__", None), str):
                owner = sys.modules.get(value.__module__)
                if owner is not None:
                    queue.append(owner)
    return list(seen.values())


def import_shas(funcs: Iterable[Callable], code_root: pathlib.Path) -> dict[str, str]:
    """Git blob shas of every module under ``code_root`` reachable from the steps' modules."""
    root = code_root.resolve()
    seen: dict[str, str] = {}
    for module in closure(funcs, root):
        path = _module_path(module, root)
        if path is not None:
            seen[path.relative_to(root).as_posix()] = git_blob_sha(path.read_bytes())
    return dict(sorted(seen.items()))


def imported_dists(funcs: Iterable[Callable], code_root: pathlib.Path) -> dict[str, str]:
    """Installed distributions the steps' closure imports, with their requirements, name to version.

    An editable install's version carries a hash of its source files, since the version does not
    move when the files do.
    """
    owners = distribution_owners()
    tops = {m.__name__.partition(".")[0] for m in closure(funcs, code_root)}
    todo = {d for top in tops for d in owners.get(top, (top,))}
    found: dict[str, str] = {}
    while todo:
        try:
            dist = importlib.metadata.distribution(todo.pop())
        except importlib.metadata.PackageNotFoundError:
            continue
        name = dist.metadata["Name"]
        if name in found:
            continue
        found[name] = dist.version + _editable_suffix(dist)
        for req in dist.requires or ():
            if "extra ==" not in req:
                todo.add(re.match(r"[A-Za-z0-9_.-]+", req).group())
    return dict(sorted(found.items()))


def _module_path(module: types.ModuleType, root: pathlib.Path) -> pathlib.Path | None:
    file = getattr(module, "__file__", None)
    if not file:
        return None
    path = pathlib.Path(file).resolve()
    if root not in path.parents or ".venv" in path.parts or not path.suffix == ".py":
        return None
    return path


@functools.cache
def distribution_owners() -> dict[str, list[str]]:
    return importlib.metadata.packages_distributions()


def _editable_suffix(dist: importlib.metadata.Distribution) -> str:
    text = dist.read_text("direct_url.json")
    info = json.loads(text) if text else {}
    if not info.get("dir_info", {}).get("editable"):
        return ""
    root = pathlib.Path(info["url"].removeprefix("file://"))
    files = [f for f in sorted(root.rglob("*.py")) if not any(p.startswith(".") for p in f.parts)]
    return "+" + content_hash(
        {f.relative_to(root).as_posix(): git_blob_sha(f.read_bytes()) for f in files}
    )
