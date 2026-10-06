"""Identity: content hashes of declarations, ULIDs, and the code and environment a fit
depends on.

A declaration is hashed by its canonical form: dataclass fields minus labels, functions
by dotted path. Code identity is separate: ``import_shas`` gives the git blob sha of
every repo module a set of functions imports, computed the way ``git hash-object`` does,
so dirty files count; ``imported_dists`` gives the installed distributions the same
closure reaches.

Examples
--------
>>> content_hash({"a": 1}) == content_hash({"a": 1})
True
>>> len(ulid())
26
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import marshal
import os
import pathlib
import re
import subprocess
import sys
import sysconfig
import time
import types
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np


class Refused(Exception):
    """The ledger refuses an operation that would break an invariant."""


HASH_LEN = 16
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


# Declarations =========================================================================


def function_ref(func: Callable) -> str:
    """The ``module:qualname`` of a module-level function; refused for a callable its
    module does not hold under that name: a lambda, a closure, a method, a partial.
    """
    module = getattr(func, "__module__", None)
    name = getattr(func, "__qualname__", repr(func))
    if getattr(sys.modules.get(module), name, None) is not func:
        raise Refused(
            f"{name!r} is not a module-level function of {module}; a declaration "
            "names functions by their dotted path"
        )
    return f"{module}:{name}"


def canonical(obj: Any) -> Any:
    """Reduce a declaration to JSON-serializable values with a stable form."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if callable(obj):
        return {"__function__": function_ref(obj)}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        module = sys.modules.get(type(obj).__module__)
        sourced = str(getattr(module, "__file__", "")).endswith(".py")
        fields = {
            f.name: canonical(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
            if not f.metadata.get("label")
            and not (sourced and _at_default(f, getattr(obj, f.name)))
        }
        kind = f"{type(obj).__module__}:{type(obj).__qualname__}"
        return {"__type__": kind, **fields}
    if isinstance(obj, dict):
        keys = {
            k if isinstance(k, str) else json.dumps([type(k).__name__, canonical(k)]): v
            for k, v in obj.items()
        }
        return {k: canonical(keys[k]) for k in sorted(keys)}
    if isinstance(obj, (list, tuple)):
        return [canonical(v) for v in obj]
    raise Refused(f"cannot serialize {type(obj).__name__} into a declaration")


def _at_default(field: dataclasses.Field, value: Any) -> bool:
    """A field holding its default is left out, so adding a defaulted field to a
    declaration keeps every existing id; only for a class whose module has a ``.py``
    source, which the memo hashes or a distribution's lock covers, so a default edited
    where the memo cannot see stays in the id.
    """
    if field.default is not dataclasses.MISSING:
        default = field.default
    elif field.default_factory is not dataclasses.MISSING:
        default = field.default_factory()
    else:
        return False
    return content_hash(value) == content_hash(default)


def content_hash(obj: Any) -> str:
    """Hex digest of the canonical form, 16 characters."""
    payload = json.dumps(canonical(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:HASH_LEN]


def bytes_hash(payload: bytes) -> str:
    """Full sha256 hex of raw bytes, the blob store key."""
    return hashlib.sha256(payload).hexdigest()


def array_hash(array: np.ndarray) -> str:
    """Full sha256 hex of an array's bytes, or of an object array's values as JSON."""
    if array.dtype.kind != "O":
        return bytes_hash(np.ascontiguousarray(array).tobytes())
    try:
        text = json.dumps(array.tolist(), ensure_ascii=False)
    except TypeError as error:
        raise Refused(
            f"an object column holds a value that is not a str, number or None: {error}"
        ) from None
    return bytes_hash(text.encode())


# Ids ==================================================================================

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


# Code identity ========================================================================


def git_blob_sha(data: bytes) -> str:
    """The sha git would give these bytes as a blob."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def repo_root(start: pathlib.Path) -> pathlib.Path:
    """The git toplevel containing ``start``, or its directory outside a repo."""
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


def imports(
    funcs: Iterable[Callable], code_root: pathlib.Path
) -> list[types.ModuleType]:
    """Every module reachable from the functions' modules through module-level names.

    Modules outside ``code_root`` are reached but not expanded, so a third-party package
    appears once and its internals are never walked, except an editable install's
    modules, which are expanded as the repo's are, their sources read once per process.
    """
    root = code_root.resolve()
    queue = [sys.modules[f.__module__] for f in funcs if f.__module__ in sys.modules]
    seen: dict[str, types.ModuleType] = {}
    while queue:
        module = queue.pop()
        if module.__name__ in seen:
            continue
        seen[module.__name__] = module
        path = _module_path(module, root)
        editable = None if path else _editable(module)
        if path or editable:
            source = path.read_bytes() if path else _editable_source(editable[1])
            names = _imported_names(source, module.__package__)
            queue.extend(sys.modules[n] for n in names if n in sys.modules)
        elif not _namespace_under(module, root):
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
    """Git blob shas of every module under ``code_root`` reachable from the
    functions' modules: what ran, as the log records it.
    """
    return {p: git_blob_sha(b) for p, b in _sources(funcs, code_root).items()}


def code_keys(funcs: Iterable[Callable], code_root: pathlib.Path) -> dict[str, str]:
    """Per module in the functions' closure, a hash of its syntax tree without
    positions, comments or layout: what the memo keys on, so a formatter pass keeps
    every id.
    """
    return {p: code_key(b) for p, b in _sources(funcs, code_root).items()}


def refuse_unseen_code(funcs: Iterable[Callable], code_root: pathlib.Path):
    """Refuse a reached module whose code the memo cannot see: a ``.py`` file outside
    ``code_root`` that no installed distribution owns, whose edit would not refit, or a
    module under it that ran from cached bytecode other than its source.

    Ownership is by the file, not the module's name: a file is the standard library's
    when it lies under its paths outside a site directory, and a distribution's when
    its record lists it, when it lies in the site directory of one that records no
    files, or when it serves an editable install, whose lock hashes the file that ran.
    A ``calendar.py`` on the path, or a fork shadowing an installed package, is
    refused.
    """
    root = code_root.resolve()
    for module in imports(funcs, root):
        file = str(getattr(module, "__file__", None) or "")
        path = pathlib.Path(file).resolve()
        if _module_path(module, root) and _stale(module):
            raise Refused(
                f"{file} ran from bytecode {module.__cached__} that is not its "
                "source, as after an edit inside one mtime second; delete the bytecode "
                "and rerun"
            )
        if (
            file.endswith(".py")
            and not path.is_relative_to(root)
            and not _stdlib(path)
            and not _editable(module)
            and not _owned(module.__name__.partition(".")[0], path)
        ):
            raise Refused(
                f"{file} is reached from the declared functions but lies outside the "
                f"code root {root} and no installed distribution owns it, so an edit "
                f"to it would not refit; move it under {root} or install it"
            )


@functools.cache
def code_key(source: bytes) -> str:
    """The code as compiled: ``ast.dump`` of the parsed source with every docstring,
    every function argument and return annotation removed and the leading import
    block sorted. Comments, layout, quote style, line numbers, docstrings, signature
    annotations and import order are out, so a function whose output reads its own
    source text, ``__doc__``, ``__annotations__`` or line numbers is outside the memo,
    as under ``python -OO``.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, _DOC_OWNERS) and (
            ast.get_docstring(node, clean=False) is not None
        ):
            del node.body[0]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.returns = None
            for arg in ast.walk(node.args):
                if isinstance(arg, ast.arg):
                    arg.annotation = None
    imported = (ast.Import, ast.ImportFrom)
    head = next(
        (i for i, n in enumerate(tree.body) if not isinstance(n, imported)),
        len(tree.body),
    )
    tree.body[:head] = sorted(tree.body[:head], key=ast.dump)
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:HASH_LEN]


_DOC_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _sources(funcs: Iterable[Callable], code_root: pathlib.Path) -> dict[str, bytes]:
    root = code_root.resolve()
    seen: dict[str, bytes] = {}
    for module in imports(funcs, root):
        path = _module_path(module, root)
        if path is not None:
            seen[path.relative_to(root).as_posix()] = path.read_bytes()
    return dict(sorted(seen.items()))


def imported_dists(
    funcs: Iterable[Callable], code_root: pathlib.Path
) -> dict[str, str]:
    """Installed distributions the functions' closure imports, with their
    requirements, name to version.

    An editable install's version carries a hash of the code keys of its modules the
    closure reaches, since the version does not move when the files do; see
    ``_editable_suffix``.
    """
    owners = distribution_owners()
    modules = imports(funcs, code_root)
    tops = {m.__name__.partition(".")[0] for m in modules}
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
        found[name] = dist.version + _editable_suffix(dist, modules)
        for req in dist.requires or ():
            if "extra ==" not in req:
                todo.add(re.match(r"[A-Za-z0-9_.-]+", req).group())
    return dict(sorted(found.items()))


def editable_roots(
    funcs: Iterable[Callable], code_root: pathlib.Path
) -> dict[str, pathlib.Path]:
    """Each editable install the functions' closure reaches, name to the git root of
    the files its modules were imported from.
    """
    roots: dict[str, pathlib.Path] = {}
    for module in imports(funcs, code_root):
        editable = _editable(module)
        if editable and editable[0] not in roots:
            roots[editable[0]] = repo_root(editable[1])
    return dict(sorted(roots.items()))


@functools.cache
def _imported_names(source: bytes, package: str | None) -> tuple[str, ...]:
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            relative = "." * node.level + (node.module or "")
            try:
                module = importlib.util.resolve_name(relative, package)
            except ImportError:
                continue
            for alias in node.names:
                sub = f"{module}.{alias.name}"
                names.append(sub if sub in sys.modules else module)
    return tuple(names)


def _namespace_under(module: types.ModuleType, root: pathlib.Path) -> bool:
    if getattr(module, "__file__", None):
        return False
    paths = getattr(module, "__path__", ())
    return any(pathlib.Path(p).resolve().is_relative_to(root) for p in paths)


def _module_path(module: types.ModuleType, root: pathlib.Path) -> pathlib.Path | None:
    file = getattr(module, "__file__", None)
    if not file:
        return None
    path = pathlib.Path(file).resolve()
    if not path.is_relative_to(root) or ".venv" in path.parts or path.suffix != ".py":
        return None
    return path


@functools.cache
def distribution_owners() -> dict[str, list[str]]:
    return importlib.metadata.packages_distributions()


def _stdlib(path: pathlib.Path) -> bool:
    unsited = {"site-packages", "dist-packages"}.isdisjoint(path.parts)
    return unsited and any(path.is_relative_to(p) for p in _stdlib_paths())


@functools.cache
def _stdlib_paths() -> tuple[pathlib.Path, ...]:
    paths = sysconfig.get_paths()
    return tuple(pathlib.Path(paths[k]).resolve() for k in ("stdlib", "platstdlib"))


def _owned(top: str, path: pathlib.Path) -> bool:
    names = distribution_owners().get(top, (top,))
    return any(p in _installed(n) for n in names for p in (path, *path.parents))


@functools.cache
def _installed(name: str) -> frozenset[pathlib.Path]:
    """The files a distribution's record lists, resolved, or its site directory when
    it records none; empty when it is not installed.
    """
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return frozenset()
    base = pathlib.Path(dist.locate_file("")).resolve()
    if dist.files is None:
        return frozenset({base})
    return frozenset(pathlib.Path(os.path.normpath(base / f)) for f in dist.files)


def _stale(module: types.ModuleType) -> bool:
    """Whether the module ran from cached bytecode that the importer took as fresh (its
    header matches the source's mtime and size) but that is not the source compiled.
    """
    try:
        data = pathlib.Path(module.__cached__).read_bytes()
        stat = pathlib.Path(module.__file__).stat()
    except (AttributeError, TypeError, OSError):
        return False
    header = [int.from_bytes(data[i : i + 4], "little") for i in (4, 8, 12)]
    fresh = [0, int(stat.st_mtime) & 0xFFFFFFFF, stat.st_size & 0xFFFFFFFF]
    if data[:4] != importlib.util.MAGIC_NUMBER or header != fresh:
        return False
    source = pathlib.Path(module.__file__).read_bytes()
    return marshal.loads(data[16:]) != compile(
        source, module.__file__, "exec", dont_inherit=True
    )


def _editable(module: types.ModuleType) -> tuple[str, pathlib.Path] | None:
    file = getattr(module, "__file__", None)
    if not file or not file.endswith(".py"):
        return None
    dist = _editable_dist(module.__name__.partition(".")[0])
    return (dist, pathlib.Path(file).resolve()) if dist else None


@functools.cache
def _editable_dist(top: str) -> str | None:
    for name in distribution_owners().get(top, (top,)):
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        if _is_editable(dist):
            return dist.metadata["Name"]
    return None


def _is_editable(dist: importlib.metadata.Distribution) -> bool:
    text = dist.read_text("direct_url.json")
    return bool(text and json.loads(text).get("dir_info", {}).get("editable"))


@functools.cache
def _editable_source(path: pathlib.Path) -> bytes:
    """An editable module's source, read once per process, so one run fits under one
    lock however the install is edited meanwhile. The repo's own modules are read
    fresh, which is how a run catches their edits.
    """
    return path.read_bytes()


def _editable_suffix(
    dist: importlib.metadata.Distribution, modules: Iterable[types.ModuleType]
) -> str:
    """``+`` and a hash of the code keys of an editable install's modules among
    ``modules``, so an edit to a module no function reaches keeps the lock; every
    module of the install when none is among them, as for one reached only as a
    requirement. Empty for an install that is not editable.
    """
    if not _is_editable(dist):
        return ""
    name = dist.metadata["Name"]
    files = {
        m.__name__: code_key(_editable_source(editable[1]))
        for m in modules
        if (editable := _editable(m)) and editable[0] == name
    }
    if files:
        return "+" + content_hash(files)
    tops = [t for t, owners in distribution_owners().items() if name in owners]
    for top in tops or [name.replace("-", "_")]:
        spec = importlib.util.find_spec(top)
        if spec is None:
            continue
        if spec.submodule_search_locations is None:
            if spec.origin and spec.origin.endswith(".py"):
                files[top] = code_key(_editable_source(pathlib.Path(spec.origin)))
            continue
        for location in spec.submodule_search_locations:
            for f in sorted(pathlib.Path(location).rglob("*.py")):
                files[f"{top}/{f.relative_to(location).as_posix()}"] = code_key(
                    _editable_source(f.resolve())
                )
    return "+" + content_hash(files)
