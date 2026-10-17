"""``lab``: the command line over the log, two verbs; reading is SQL over the views.

Declarations are plain module attributes, so a project may organize them freely: one
script, or ``dataset.py`` + ``functions.py`` + ``experiment.py``, or a file per idea.
``lab ingest`` calls ``dataset(ledger, *args)`` from a module, the extra arguments as
strings, and prints the id it returns. ``lab run`` takes one or
more modules, reads ``pipelines`` (a list of ``Pipeline``) from each and
``evaluations`` (a list of ``Evaluation``, or a function of the dataset id returning
one) from exactly one of them, so a file an agent wrote holding only new pipelines
runs beside the project's declarations; every pipeline is scored under every
evaluation. The code root the memo hashes from is the git root of the module that
declares the evaluations, and both verbs refuse a module outside a git repository; the
modules themselves join no id, so adding a pipeline to
one refits nothing else. A module is a dotted name importable from the current
directory or a ``.py`` path, imported without writing bytecode. The dataset defaults
to the newest one recorded when the ledger holds one source. The ledger root comes
from ``--root``, before or after the verb, or ``ML_LAB_ROOT`` (default ``.ml-lab``);
only ``lab ingest`` creates one, and ``lab run`` refuses a root without a ledger. The
actor comes from ``ML_LAB_ACTOR``; both verbs refuse it unset.

Examples
--------
$ export ML_LAB_ROOT=$PWD/ledger ML_LAB_ACTOR=daniel
$ lab ingest project/dataset.py 20
$ lab run project/experiment.py
$ lab run project/experiment.py ideas/agent7.py --dry-run
$ sqlite3 -box $ML_LAB_ROOT/ml_lab.sqlite "SELECT * FROM board"
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import os
import pathlib
import sys
from typing import Any

from ml_lab import dataset, identity, runs
from ml_lab.experiment import Evaluation, Pipeline
from ml_lab.ledger import Ledger, Refused, actor, note

DEFAULT_ROOT = ".ml-lab"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lab")
    parser.add_argument("--root", default=os.environ.get("ML_LAB_ROOT", DEFAULT_ROOT))
    rooted = argparse.ArgumentParser(add_help=False)
    rooted.add_argument("--root", default=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "ingest",
        parents=[rooted],
        help="call the module's dataset(ledger, *args); prints the id",
    )
    p.add_argument("module", help="module name or .py path exposing dataset()")
    p.add_argument("args", nargs="*")
    p.set_defaults(handler=_ingest)

    p = sub.add_parser(
        "run", parents=[rooted], help="fit, predict and score every declared pipeline"
    )
    p.add_argument(
        "experiments", nargs="+", help="modules or .py paths exposing pipelines"
    )
    p.add_argument(
        "--dataset",
        help="dataset id prefix or source name; default: the newest, when the ledger "
        "holds one source",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="write nothing; say what a run would compute and why, computing and "
        "probing a new feature set on its first span so its ids are real",
    )
    p.set_defaults(handler=_run)

    p = sub.add_parser(
        "note", parents=[rooted], help="record an idea, decision or step as a note"
    )
    p.add_argument("text")
    p.add_argument(
        "--about",
        nargs="*",
        default=[],
        help="ids the note is about, as lab run prints",
    )
    p.add_argument(
        "--tag", default="note", help="its kind: note, idea, decision, refit"
    )
    p.set_defaults(handler=_note)

    args = parser.parse_args(argv)
    root = pathlib.Path(args.root)
    try:
        actor()
        if root.exists() and not root.is_dir():
            raise Refused(f"{root} is not a directory")
        if args.command != "ingest" and not (root / "ml_lab.sqlite").exists():
            raise Refused(
                f"no ledger at {root.resolve()}; set ML_LAB_ROOT or --root, or lab "
                "ingest to create one"
            )
        return args.handler(args, Ledger(root))
    except (Refused, KeyError, ImportError, SyntaxError) as refused:
        print(f"lab {args.command}: {refused}", file=sys.stderr)
        return 1


# Handlers =============================================================================


def _ingest(args: argparse.Namespace, ledger: Ledger) -> int:
    module = _load(args.module)
    ingest = getattr(module, "dataset", None)
    if ingest is None:
        raise Refused(f"{args.module} declares no dataset(ledger, *args)")
    _code_root(module)
    print(ingest(ledger, *args.args))
    return 0


def _note(args: argparse.Namespace, ledger: Ledger) -> int:
    print(note(ledger, args.text, about=args.about, tag=args.tag))
    return 0


def _run(args: argparse.Namespace, ledger: Ledger) -> int:
    """Every evaluation in turn, one shared plan; a pipeline that fails in one is
    skipped in the rest of this run and named once at the end, since failures are
    not memoized and a rerun retries it.
    """
    pipelines, evaluations, modules = _experiments(args, ledger)
    home = next(m for m in modules if hasattr(m, "evaluations"))
    root = _code_root(home)
    failed: dict[str, str] = {}
    failed_ids: set[str] = set()
    planned: set[str] = set()
    total = runs.RunReport()
    for pipeline in pipelines:
        print(f"pipeline {pipeline.id} {pipeline.name}".rstrip())
    for evaluation in evaluations:
        print(f"evaluation {evaluation.id} {evaluation.name}".rstrip())
        live = [p for p in pipelines if p.id not in failed_ids]
        if pipelines and not live:
            print("skipped: every pipeline failed above")
            continue
        report = runs.run(
            ledger,
            live,
            evaluation,
            code_root=root,
            log=lambda line: print(line, flush=True),
            dry=args.dry_run,
            planned=planned,
        )
        for name, error in report.failed.items():
            print(f"lab run: {name} failed: {error}")
        failed |= report.failed
        failed_ids |= report.failed_ids
        for field in dataclasses.fields(runs.RunReport):
            value = getattr(report, field.name)
            if isinstance(value, int):
                vars(total)[field.name] += value
        if args.dry_run:
            print(f"dry run: would compute {_counts(report)}; reuse {_reuse(report)}")
        elif not report.run:
            print(
                f"up to date: {report.scores_reused} scores; {report.features_reused} "
                f"feature sets, {report.fits_reused} fits and "
                f"{report.predictions_reused} predictions reused, nothing written"
            )
        else:
            print(f"run {report.run}: {_counts(report)}")
    if len(evaluations) > 1:
        verb = "would compute" if args.dry_run else "computed"
        print(
            f"total over {len(evaluations)} evaluations: {verb} {_counts(total)}; "
            f"reuse {_reuse(total)}"
        )
    if args.dry_run and total.features_computed:
        print(
            "dry run: a new feature set is computed and probed on its first span "
            "only, or taken from an earlier dry run; the full probes and those of "
            "predict and postprocess run only in the real run"
        )
    if failed:
        print(f"failed, skipped in later evaluations: {', '.join(failed)}")
    return 1 if failed else 0


# Private Functions ====================================================================


def _counts(report: runs.RunReport) -> str:
    """The computed counts of a report as one phrase, features first when any."""
    features = f"features {report.features_computed}, " if _featured(report) else ""
    return (
        f"{features}fits {report.fits_computed}, predictions "
        f"{report.predictions_computed}, scores {report.scores_recorded}"
    )


def _featured(report: runs.RunReport) -> bool:
    """Whether the report touched a feature set, so its counts name features."""
    return bool(report.features_computed or report.features_reused)


def _reuse(report: runs.RunReport) -> str:
    """The reused counts of a report in the order of ``_counts``."""
    features = f"{report.features_reused}, " if _featured(report) else ""
    return (
        f"{features}{report.fits_reused}, {report.predictions_reused}, "
        f"{report.scores_reused}"
    )


def _experiments(
    args: argparse.Namespace, ledger: Ledger
) -> tuple[list[Pipeline], list[Evaluation], list[Any]]:
    """Pipelines from every module named; the evaluations from the one module that
    declares them; the modules. Two names on one evaluation, or one name on two, are
    refused.
    """
    modules = [_load(spec) for spec in args.experiments]
    for m in modules:
        for attr in ("pipelines", "evaluations"):
            held = getattr(m, attr, [])
            if not (isinstance(held, (list, tuple)) or callable(held)):
                raise Refused(
                    f"{m.__name__}.{attr} is a {type(held).__name__}, not a list"
                )
    pipelines = [p for m in modules for p in getattr(m, "pipelines", [])]
    found = [m.evaluations for m in modules if hasattr(m, "evaluations")]
    if len(found) != 1:
        raise Refused(
            f"{len(found)} modules declare evaluations across {args.experiments}; "
            "need one"
        )
    evaluations = found[0]
    if callable(evaluations):
        evaluations = evaluations(_dataset(ledger, args.dataset))
    evaluations = [
        dataclasses.replace(e, dataset_id=dataset.resolve(ledger, e.dataset_id))
        for e in evaluations
    ]
    if args.dataset and not callable(found[0]):
        chosen = _dataset(ledger, args.dataset)
        off = [e.name or e.id for e in evaluations if e.dataset_id != chosen]
        if off:
            raise Refused(
                f"evaluations {off} are declared on another dataset than --dataset "
                f"{args.dataset} ({chosen[:8]}); a list of evaluations fixes its own"
            )
    names, ids = {e.name or e.id for e in evaluations}, {e.id for e in evaluations}
    if not (evaluations and len(names) == len(ids) == len(evaluations)):
        raise Refused(
            f"{len(evaluations)} evaluations, {len(names)} names, {len(ids)} ids; one "
            "name per evaluation and one evaluation per name"
        )
    return pipelines, evaluations, modules


def _code_root(module: Any) -> pathlib.Path:
    """The git root of the module's file, refused outside a repo: a run records the
    commit its code came from.
    """
    root = identity.repo_root(pathlib.Path(module.__file__))
    if not (root / ".git").exists():
        raise Refused(
            f"{module.__file__} is not inside a git repository; git init the project, "
            "since provenance names a commit"
        )
    return root


def _dataset(ledger: Ledger, selector: str | None) -> str:
    """The dataset named by an id prefix or a source; with no selector, the newest,
    provided every dataset in the ledger shares one source. A source written by
    several actors resolves to the newest dataset the running actor recorded under
    it, and is refused with every actor's id when the running actor recorded none,
    since the newest would silently be someone else's.
    """
    rows = ledger.sql("SELECT id, source, actor FROM event_dataset ORDER BY seq")
    by_source: dict[str, str] = {r["source"]: r["id"] for r in rows}
    if selector is None and len(by_source) == 1:
        selector = rows[-1]["source"]
    named = [r for r in rows if r["source"] == selector]
    if len({r["actor"] for r in named}) > 1:
        own = [r for r in named if r["actor"] == actor()]
        if not own:
            raise Refused(
                f"source {selector} names datasets by several actors, none by "
                f"{actor()}; pass an id: "
                + ", ".join(f"{r['id'][:8]} ({r['actor']})" for r in named)
            )
        return own[-1]["id"]
    if selector in by_source:
        return by_source[selector]
    if selector:
        return dataset.resolve(ledger, selector)
    if len(by_source) > 1:
        choices = ", ".join(f"{s} ({i[:8]})" for s, i in by_source.items())
        raise Refused(f"ledger holds several sources, pass --dataset: {choices}")
    if not rows:
        raise Refused(
            f"dataset (newest): 0 matches in {ledger.root.resolve()}; lab ingest first"
        )
    return rows[-1]["id"]


def _load(spec: str) -> Any:
    """Import a dotted name from the current directory, or a .py path by its package
    name, writing no bytecode, so a later run never reads a stale cache.
    """
    sys.dont_write_bytecode = True
    if spec.endswith(".py"):
        path = spec_path = pathlib.Path(spec).resolve()
        parts = [path.stem]
        while (path.parent / "__init__.py").exists():
            path = path.parent
            parts.insert(0, path.name)
        sys.path.insert(0, str(path.parent))
        spec = ".".join(parts)
        loaded = getattr(sys.modules.get(spec), "__file__", None)
        if loaded and pathlib.Path(loaded).resolve() != spec_path:
            raise Refused(
                f"{spec_path} and {loaded} both import as {spec}; put them in packages "
                "(a directory with __init__.py) or rename one"
            )
    else:
        sys.path.insert(0, os.getcwd())
    return importlib.import_module(spec)


if __name__ == "__main__":
    raise SystemExit(main())
