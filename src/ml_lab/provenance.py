"""The facts a run records about where it ran: git state, resolution, host.

None of these enter an id; they say what produced a fit. A fact that cannot be read
is recorded as far as it was read, never aborting the run.

Examples
--------
>>> host()["cpu_count"]  # doctest: +SKIP
8
"""

from __future__ import annotations

import os
import pathlib
import platform
import re
import shutil
import subprocess
import tempfile
from typing import Any

from ml_lab import formats
from ml_lab.ledger import Ledger

# Public Functions ===================================================================


UNTRACKED_DIFF_CAP = 1 << 20


EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def git_state(ledger: Ledger, root: pathlib.Path) -> dict[str, Any]:
    """The commit, whether the tree is dirty, and one diff of every change against
    HEAD, untracked files included through a temporary index so git runs once; an
    untracked file over a mebibyte is named, not diffed.
    """

    def git(*args: str, env: dict[str, str] | None = None) -> bytes | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=root, capture_output=True, check=True, env=env
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return None

    def text(out: bytes | None) -> str:
        return (out or b"").decode(errors="surrogateescape")

    git_index = git("rev-parse", "--git-path", "index")
    if git_index is None:
        return {"commit": None, "dirty": None, "diff": None}
    head = text(git("rev-parse", "--verify", "-q", "HEAD")).strip() or None
    dirty = bool(text(git("status", "--porcelain")).strip())
    diff = None
    if dirty:
        paths = text(git("ls-files", "-z", "--others", "--exclude-standard")).split(
            "\0"
        )
        big = {
            p for p in paths if p and (root / p).lstat().st_size > UNTRACKED_DIFF_CAP
        }
        small = [p for p in paths if p and p not in big]
        with tempfile.TemporaryDirectory() as tmp:
            index = pathlib.Path(tmp) / "index"
            current = root / text(git_index).strip()
            if current.is_file():
                shutil.copy(current, index)
            env = {**os.environ, "GIT_INDEX_FILE": str(index)}
            for at in range(0, len(small), 1000):
                git("add", "-N", "--", *small[at : at + 1000], env=env)
            patch = (git("diff", head or EMPTY_TREE, env=env) or b"") + "".join(
                f"# untracked, over {UNTRACKED_DIFF_CAP >> 20} MiB, not diffed: {p}\n"
                for p in big
            ).encode(errors="surrogateescape")
        diff = {"sha": ledger.put_blob(patch), "format": formats.Format.DIFF}
    return {"commit": head, "dirty": dirty, "diff": diff}


def lock_text(dists: dict[str, str]) -> str:
    lines = [
        f"python=={platform.python_version()}",
        *(f"{n}=={v}" for n, v in dists.items()),
    ]
    return "\n".join(lines) + "\n"


def resolution(ledger: Ledger, root: pathlib.Path) -> dict[str, str] | None:
    for path in [root / "uv.lock", *sorted(root.glob("requirements*.txt"))]:
        if path.exists():
            return {"path": path.name, **text_blob(ledger, path.read_text())}
    return None


def text_blob(ledger: Ledger, text: str) -> dict[str, str]:
    return {"sha": ledger.put_blob(text.encode()), "format": formats.Format.TEXT}


def host() -> dict[str, Any]:
    """Host facts for a run.

    The machine, and its NVIDIA GPUs with the driver and CUDA versions when
    ``nvidia-smi`` answers, so a GPU fit's provenance names the device that produced
    it. Facts never abort a run: a device the parser cannot read is recorded as far
    as it was read.
    """
    facts = {
        "hostname": platform.node(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count() or 1,
        "python": platform.python_version(),
    }
    try:
        return {**facts, **_gpus()}
    except Exception as error:  # noqa: BLE001
        return {**facts, "gpu_error": repr(error)}


COMMAND_TIMEOUT_S = 5


# Private Functions ==================================================================


def _gpus() -> dict[str, Any]:
    """The GPUs and CUDA version ``nvidia-smi`` reports; nothing without it."""
    query = _command(
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    )
    if query is None:
        return {}
    gpus = []
    for line in query.splitlines():
        parts = [p.strip() for p in line.strip().rstrip(",").rsplit(",", 2)]
        if len(parts) == 3:
            name, driver, memory = parts
            gpus.append(
                {
                    "name": name,
                    "driver": driver,
                    "memory_mb": int(memory) if memory.isdigit() else None,
                }
            )
    summary = _command("nvidia-smi") or ""
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", summary)
    return {"gpu": gpus, "cuda": cuda.group(1) if cuda else None}


def _command(*args: str) -> str | None:
    """A command's stdout, or None when it is missing, fails or hangs."""
    try:
        out = subprocess.run(
            args, capture_output=True, text=True, check=True, timeout=COMMAND_TIMEOUT_S
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return out.stdout
