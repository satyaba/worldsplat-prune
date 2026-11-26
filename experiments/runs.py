"""Declarative manifest of the runs to execute.

One place that says *what* each run is (method, iterations, dataset, extra
args); :mod:`experiments.runner` says *how* to prepare and launch it. Add a
:class:`RunSpec` to :data:`RUNS` and it becomes runnable via the CLI or
``run_all()`` from a notebook.

    python -m experiments.runs --list
    python -m experiments.runs --run base_30k
    python -m experiments.runs --all

Dataset layout assumed under ``$HOME_DIR/Dataset/<dataset>``::

    <dataset>/images          # RGB (rgb_<N>.png for Replica)
    <dataset>/sparse/0/*.bin  # COLMAP
    <dataset>/semantic_class  # uint8 class-id PNGs (band/interior eval)

The local method id is ``worldsplat`` (this checkout). The training-time
density-control components of the old SGDC line have been removed. This module
holds two manifests: :data:`RUNS` (the training baselines) and
:data:`PRUNE_RUNS` (the post-reconstruction pruning-ablation arms of §4–5).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from experiments.runner import run_method, _split_extra, REPO_ROOT

DEFAULT_DATASET = "office_0/Sequence_1"
DEFAULT_ITERATIONS = 30_000


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    method: str
    iterations: int = DEFAULT_ITERATIONS
    dataset: str = DEFAULT_DATASET
    images: Optional[str] = None
    extra: tuple = ()
    sanity: bool = False                    # short smoke test (not a result run)
    notes: str = ""


# --------------------------------------------------------------------------- #
# RUNS
# --------------------------------------------------------------------------- #
# Training baselines. All runs use the shared `--eval` split (llffhold=8) so
# rows are comparable across methods.
RUNS: Dict[str, RunSpec] = {
    r.run_id: r
    for r in (
        RunSpec(
            run_id="vanilla_30k",
            method="vanilla_3dgs",
            iterations=30_000,
            notes="Upstream graphdeco 3DGS (stock rasterizer), --eval split.",
        ),
        RunSpec(
            run_id="base_30k",
            method="worldsplat_prune",
            iterations=30_000,
            notes="This checkout (base 3DGS trainer), --eval split.",
        ),
        # --- sanity runs: same configs at 3_000 iters (cross the first
        # densification + opacity reset, then save). ---
        RunSpec(
            run_id="vanilla_sanity",
            method="vanilla_3dgs",
            iterations=3_000,
            sanity=True,
            notes="SANITY: upstream 3DGS @3k.",
        ),
        RunSpec(
            run_id="base_sanity",
            method="worldsplat_prune",
            iterations=3_000,
            sanity=True,
            notes="SANITY: this checkout @3k.",
        ),
    )
}


def full_runs() -> List[RunSpec]:
    return [r for r in RUNS.values() if not r.sanity]


def sanity_runs() -> List[RunSpec]:
    return [r for r in RUNS.values() if r.sanity]


# --------------------------------------------------------------------------- #
# PRUNE_RUNS — post-reconstruction pruning abalations (§4-5)
# --------------------------------------------------------------------------- #
# Each spec is one pruning pass over a *trained* base model at
# ``<home>/output/<base>``. The arms are the control matrix (§5.2):
#   semantic  band protect + purity confidence            (the candidate)
#   geometry  neither (geometry-only score, no band)      (the control)
#   random    random allocation, band off                 (structure bar)
#   noband    confidence, no band protect                 (ablation)
#   seg       per-segment allocation                      (ablation, §7.3)
# Every arm at a given (keep_frac, seed) yields the same Gaussian count, so rows
# are matched. Scored through the same eval_worldsplat path (metrics.py).
PRUNE_ARMS = ("semantic", "geometry", "random", "noband", "seg")
PRUNE_SEEDS = (0, 1, 2)
PRUNE_KEEP_FRACS = (0.25, 0.5, 0.75)
PRUNE_STRENGTHS = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class PruneSpec:
    run_id: str
    base: str = "base_30k"               # trained model id under <home>/output
    arm: str = "semantic"
    keep_frac: float = 0.5
    strength: float = 0.5
    curve: float = 1.0
    band: float = 0.04
    epsilon: float = 0.005
    allocation: str = "global"
    target: Optional[int] = None         # mode (b): instance id kept whole
    seed: int = 0
    label_adapter: str = "composite"
    label_dir: Optional[str] = None
    notes: str = ""


def _prune_id(base: str, arm: str, keep_frac: float, strength: float, seed: int) -> str:
    return f"p_{base}_{arm}_k{keep_frac:g}_s{strength:g}_{seed}"


def prune_specs(*, base: str = "base_30k",
                seeds: Sequence[int] = PRUNE_SEEDS,
                keep_fracs: Sequence[float] = PRUNE_KEEP_FRACS,
                strengths: Sequence[float] = PRUNE_STRENGTHS,
                label_adapter: str = "composite",
                label_dir: Optional[str] = None) -> List[PruneSpec]:
    """Build the pruning grid: 3 arms x keep_fracs x seeds, plus a `strength`
    sweep of the candidate and the two ablations (deduplicated by run_id)."""
    specs: List[PruneSpec] = []

    def add(arm, keep_frac, strength, seed, **kw):
        specs.append(PruneSpec(
            _prune_id(base, arm, keep_frac, strength, seed), base=base, arm=arm,
            keep_frac=keep_frac, strength=strength, seed=seed,
            label_adapter=label_adapter, label_dir=label_dir, **kw))

    for arm in ("semantic", "geometry", "random"):
        for kf in keep_fracs:
            for seed in seeds:
                add(arm, kf, 0.5, seed)
    # strength sweep of the candidate at the primary budget (§5.1 order: s first)
    for st in strengths:
        for seed in seeds:
            add("semantic", 0.5, st, seed)
    # ablations at the primary budget
    for seed in seeds:
        add("noband", 0.5, 0.5, seed)
        add("seg", 0.5, 0.5, seed, allocation="per_segment")

    seen, out = set(), []
    for s in specs:
        if s.run_id not in seen:
            seen.add(s.run_id)
            out.append(s)
    return out


PRUNE_RUNS: Dict[str, PruneSpec] = {s.run_id: s for s in prune_specs()}


def _arm_flags(spec: PruneSpec) -> List[str]:
    """The flat `prune_model.py` flags that select an arm (§5.2)."""
    if spec.arm == "geometry":
        return ["--no_band", "--no_purity"]
    if spec.arm == "random":
        return ["--random"]                       # driver: random => band off
    if spec.arm == "noband":
        return ["--no_band"]
    if spec.arm == "seg":
        return ["--allocation", "per_segment"]
    if spec.arm == "semantic":
        return []
    raise ValueError(f"unknown prune arm {spec.arm!r}; known: {PRUNE_ARMS}")


def build_prune_cmd(spec: PruneSpec, base_dir, out_dir,
                    python: str = sys.executable) -> List[str]:
    """The `prune_model.py` command line for one pruning spec."""
    cmd = [python, "prune_model.py", "-m", str(base_dir),
           "--label_adapter", spec.label_adapter,
           "--keep_frac", str(spec.keep_frac),
           "--strength", str(spec.strength),
           "--curve", str(spec.curve),
           "--band", str(spec.band),
           "--epsilon", str(spec.epsilon),
           "--allocation", spec.allocation,
           "--seed", str(spec.seed),
           "--out_dir", str(out_dir)]
    if spec.label_dir:
        cmd += ["--label_dir", spec.label_dir]
    if spec.target is not None:
        cmd += ["--target", str(spec.target)]
    return cmd + _arm_flags(spec)


def prune_dir(spec: PruneSpec, home_dir: Optional[str] = None) -> Path:
    home = home_dir or os.environ.get("HOME_DIR")
    if not home:
        raise RuntimeError("set $HOME_DIR or pass home_dir")
    return Path(home) / "output" / spec.run_id


def run_prune(spec: PruneSpec, home_dir: Optional[str] = None,
              dry: bool = False) -> Path:
    """Prune ``spec``'s base model into ``<home>/output/<spec.run_id>``."""
    home = home_dir or os.environ.get("HOME_DIR")
    if not home:
        raise RuntimeError("set $HOME_DIR or pass home_dir")
    base_dir = Path(home) / "output" / spec.base
    out_dir = prune_dir(spec, home_dir)
    print(f"[runs] prune {spec.run_id}: base={base_dir} arm={spec.arm} "
          f"keep_frac={spec.keep_frac:g} strength={spec.strength:g} seed={spec.seed}")
    cmd = build_prune_cmd(spec, base_dir, out_dir)
    print(f"[runs]   $ {' '.join(cmd)}")
    if not dry:
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    return out_dir


def run_prune_by_id(run_id: str, **kw) -> Path:
    if run_id not in PRUNE_RUNS:
        raise KeyError(f"unknown prune run {run_id!r}; known: {sorted(PRUNE_RUNS)}")
    return run_prune(PRUNE_RUNS[run_id], **kw)


def run_prune_all(**kw) -> None:
    for spec in PRUNE_RUNS.values():
        run_prune(spec, **kw)


def extract_prune_run(run_id: str, *, csv_path=None, home_dir: Optional[str] = None,
                      **kw) -> Optional[dict]:
    """Score one pruning run through the shared metric path (eval_worldsplat)."""
    from experiments.metrics import extract_one

    spec = PRUNE_RUNS[run_id]
    return extract_one(prune_dir(spec, home_dir), iteration=-1, run_id=run_id,
                       name=spec.notes or spec.run_id, csv_path=csv_path,
                       label_adapter=spec.label_adapter, label_dir=spec.label_dir,
                       **kw)


def extract_prune_all(*, csv_path=None, home_dir: Optional[str] = None,
                      **kw) -> List[Optional[dict]]:
    return [extract_prune_run(rid, csv_path=csv_path, home_dir=home_dir, **kw)
            for rid in PRUNE_RUNS]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def resolve_source(spec: RunSpec, home_dir: Optional[str] = None,
                   dataset_root: Optional[str] = None) -> Path:
    dataset_root = dataset_root or os.environ.get("DATASET_ROOT")
    if dataset_root:
        return Path(dataset_root) / spec.dataset
    home = home_dir or os.environ.get("HOME_DIR")
    if not home:
        raise RuntimeError("set $HOME_DIR or pass dataset_root=/home_dir=")
    return Path(home) / "Dataset" / spec.dataset


def _model_dir(spec: RunSpec, home_dir: Optional[str], tag: Optional[str] = None) -> Path:
    home = home_dir or os.environ.get("HOME_DIR")
    if not home:
        raise RuntimeError("set $HOME_DIR or pass home_dir")
    name = spec.run_id if not tag else f"{spec.run_id}_{tag}"
    return Path(home) / "output" / name


def _extra_args(spec: RunSpec, source: Path) -> List[str]:
    return list(spec.extra)


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
def run(spec: RunSpec, home_dir: Optional[str] = None, dataset_root: Optional[str] = None,
        workdir_root: str = "/content", force: bool = False,
        skip_prepare: bool = False, dry: bool = False,
        extra: Union[str, Sequence[str], None] = None,
        tag: Optional[str] = None, model_dir: Optional[str] = None,
        dataset: Optional[str] = None) -> Path:
    """Launch one run.

    ``extra`` adds CLI tokens on top of the spec's own ``extra`` (appended AFTER,
    so argparse's last-wins lets them override). ``tag``/``model_dir`` give the
    call its own output directory; ``dataset`` overrides the spec's dataset path.
    """
    if dataset:
        spec = replace(spec, dataset=dataset)
    source = resolve_source(spec, home_dir, dataset_root)
    if model_dir is None:
        model_dir = _model_dir(spec, home_dir, tag=tag)
    effective_id = spec.run_id if not tag else f"{spec.run_id}_{tag}"
    overrides = _split_extra([extra] if isinstance(extra, str) else extra)
    print(f"[runs] {effective_id}: method={spec.method} iters={spec.iterations} "
          f"source={source}")
    if spec.notes:
        print(f"[runs]   note: {spec.notes}")
    if overrides:
        print(f"[runs]   overrides: {' '.join(overrides)}")
    return run_method(
        spec.method,
        source=source,
        model_dir=model_dir,
        workdir_root=workdir_root,
        iterations=spec.iterations,
        images=spec.images,
        extra=_extra_args(spec, source) + overrides,
        force=force,
        skip_prepare=skip_prepare,
        dry=dry,
    )


def run_by_id(run_id: str, **kw) -> Path:
    try:
        spec = RUNS[run_id]
    except KeyError:
        raise KeyError(f"unknown run {run_id!r}; known: {sorted(RUNS)}")
    return run(spec, **kw)


def run_all(**kw) -> None:
    """Run every non-sanity run in the manifest."""
    for spec in full_runs():
        run(spec, **kw)


def run_sanity(**kw) -> None:
    """Run the short sanity runs only."""
    for spec in sanity_runs():
        run(spec, **kw)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run the experiment manifest.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="list runs and exit")
    g.add_argument("--run", metavar="RUN_ID", help="run one run id")
    g.add_argument("--all", action="store_true", help="run every non-sanity run")
    g.add_argument("--sanity", action="store_true", help="run the short sanity runs only")
    g.add_argument("--list-prune", action="store_true",
                   help="list the pruning-ablation runs and exit")
    g.add_argument("--prune", metavar="RUN_ID", help="run one pruning-ablation id")
    g.add_argument("--all-prune", action="store_true",
                   help="run every pruning-ablation run")
    p.add_argument("--home-dir", default=os.environ.get("HOME_DIR"))
    p.add_argument("--dataset-root", default=os.environ.get("DATASET_ROOT"),
                   help="override the root above per-run <dataset> folders")
    p.add_argument("--workdir-root", default="/content")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dataset", default=None,
                   help="override the dataset path of --run (relative to --dataset-root)")
    args = p.parse_args(argv)

    if args.list:
        for r in RUNS.values():
            tag = "sanity" if r.sanity else "full"
            print(f"[{tag:6s}] {r.run_id:20s} method={r.method:13s} iters={r.iterations:<6d} "
                  f"dataset={r.dataset}  {r.notes}")
        return 0

    if args.list_prune:
        for r in PRUNE_RUNS.values():
            print(f"[prune] {r.run_id:30s} base={r.base:12s} arm={r.arm:9s} "
                  f"keep_frac={r.keep_frac:<4g} strength={r.strength:<4g} seed={r.seed}")
        return 0
    if args.prune:
        run_prune_by_id(args.prune, home_dir=args.home_dir, dry=args.dry_run)
        return 0
    if args.all_prune:
        run_prune_all(home_dir=args.home_dir, dry=args.dry_run)
        return 0

    kw = dict(home_dir=args.home_dir, dataset_root=args.dataset_root,
              workdir_root=args.workdir_root, force=args.force,
              skip_prepare=args.skip_prepare, dry=args.dry_run)
    if args.sanity:
        run_sanity(**kw)
    elif args.all:
        run_all(**kw)
    else:
        run_by_id(args.run, dataset=args.dataset, **kw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
