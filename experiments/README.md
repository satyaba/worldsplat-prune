# experiments/ — cross-codebase experiment harness (WorldSplat-Prune)

Run **this** codebase (WorldSplat-Prune) and **external** 3DGS codebases
(upstream graphdeco 3DGS, and future baselines) through one interface. The
harness handles cloning, pinning, CUDA-extension install, and — importantly —
**swapping the rasterizer** between methods.

## Why rasterizer swapping matters

Every 3DGS codebase installs its CUDA rasterizer under the same import name,
`diff_gaussian_rasterization`:

- **WorldSplat-Prune** pins the `dr_aa` (antialiasing) branch of the rasterizer.
- **Upstream graphdeco** uses its default (`main`) rasterizer.

They cannot coexist in one Python runtime. Each method's `install` first
`pip uninstall`s the package, and the runner re-installs only when you switch
methods (tracked by `/content/.worldsplat_rasterizer_method`).

> **Segmentation note.** There is **no seg-aware rasterizer** here. Label maps
> are read from disk (through the `worldsplat_prune` adapters) and segmentation
> channels are never rendered, so the RGB rasterizer is stock for every method.

## Quick start (Colab / Linux)

```bash
# from the WorldSplat-Prune repo root
export HOME_DIR=/content/drive/MyDrive/Magister/Thesis/Script
export DATASET_DIR=$HOME_DIR/Dataset/office_0/Sequence_1
```

### The experiment manifest

`experiments/runs.py` declares the runs; one command executes them:

```bash
python -m experiments.runs --list             # show runs
python -m experiments.runs --sanity           # short smoke runs (verify the pipeline)
python -m experiments.runs --run base_30k     # one run
python -m experiments.runs --all              # every non-sanity run
```

Runs currently declared (training baselines; the pruning-method ablation grid is
in the separate `PRUNE_RUNS` manifest below):

| run id | method | iters | notes |
|---|---|---|---|
| `vanilla_30k` | `vanilla_3dgs` | 30000 | upstream graphdeco 3DGS (stock rasterizer), `--eval` |
| `base_30k` | `worldsplat_prune` | 30000 | this checkout (base 3DGS trainer), `--eval` |
| `vanilla_sanity` | `vanilla_3dgs` | 3000 | SANITY smoke test |
| `base_sanity` | `worldsplat_prune` | 3000 | SANITY: base trainer @3k |

The **sanity runs** reuse the full-run configs at 3 000 iterations — enough to
cross the first densification (500), hit the opacity reset (3000), and save. Use
them to verify the whole pipeline before committing to 30 000 iters. `--all` and
`extract_all()` skip sanity runs; `--sanity` runs only them.

Add a run by appending a `RunSpec` to `RUNS`. All runs share the `--eval`
(llffhold=8) train/test split, so rows are comparable across methods.

### The pruning-ablation manifest (§4–5)

Post-reconstruction pruning has its own grid (`PRUNE_RUNS`): each spec prunes a
**trained base model** (`<home>/output/<base>`, default `base_30k`) and is scored
through the same metric path. The arms are the §5.2 control matrix.

```bash
python -m experiments.runs --list-prune        # show the grid
python -m experiments.runs --prune p_base_30k_semantic_k0.5_s0.5_0
python -m experiments.runs --all-prune
```

| arm | `prune_model.py` flags | role |
|---|---|---|
| `semantic` | _(defaults)_ | candidate: band protect + purity confidence |
| `geometry` | `--no_band --no_purity` | control: geometry-only score, no band |
| `random` | `--random` | structure bar (random allocation, band off) |
| `noband` | `--no_band` | ablation: confidence, no band protect |
| `seg` | `--allocation per_segment` | ablation (§7.3) |

Default grid: 3 arms × `keep_frac ∈ {0.25, 0.5, 0.75}` × seeds `{0, 1, 2}`, plus
a `strength` sweep of the candidate and the two ablations. Every arm at a given
`(keep_frac, seed)` keeps the **same Gaussian count**, so rows are matched.
Regenerate a narrower grid with
`prune_specs(keep_fracs=..., strengths=..., seeds=...)`.

From a notebook:

```python
import os
from experiments.runs import run_prune_all, extract_prune_all

os.environ["HOME_DIR"] = "/content/drive/MyDrive/Magister/Thesis/Script"
run_prune_all()                                   # writes <home>/output/p_...
extract_prune_all(csv_path=os.environ["HOME_DIR"] +
                  "/output/prune_metrics.csv")     # one CSV row per arm
```

Score a one-off arm without the manifest:

```bash
python prune_model.py -m "$HOME_DIR/output/base_30k" --label_adapter composite \
    --keep_frac 0.5 --out_dir "$HOME_DIR/output/my_arm"
python -m experiments.metrics --model-dir "$HOME_DIR/output/my_arm" \
    --label-adapter composite
```

### Running one method directly

```bash
# upstream 3DGS (30k, eval split)
python -m experiments.runner --method vanilla_3dgs --run-id runB1 \
    --iterations 30000

# this checkout (same 30k)
python -m experiments.runner --method worldsplat_prune --run-id ext \
    --iterations 30000
```

`--source` and `--model-dir` default to `$DATASET_DIR` and
`$HOME_DIR/output/$RUN_ID`. Add `--dry-run` to print commands without executing.

### From a notebook cell

```python
import os
from experiments.runs import run_by_id, RUNS

os.environ["HOME_DIR"] = "/content/drive/MyDrive/Magister/Thesis/Script"
os.environ["DATASET_ROOT"] = os.environ["HOME_DIR"] + "/Dataset"

for run_id in RUNS:            # or run_by_id("base_30k")
    run_by_id(run_id)
```

### Per-run hyperparameter overrides (notebook)

`run_by_id` (and `run`) accept an `extra` argument — CLI tokens appended **after**
the spec's own args, so argparse's last-wins lets them override any
hyperparameter for a single call without editing the manifest:

```python
run_by_id("base_30k", extra="--position_lr_init 0.00016")
run_by_id("base_30k", extra=["--densify_until_iter", "10000"])
run_by_id("base_30k", extra="--lambda_dssim 0.3", dry=True)   # preview
```

**Sweeps need distinct output dirs.** `tag` appends a suffix to the run id
(`<home>/output/<run_id>_<tag>`), and `model_dir` overrides the path outright:

```python
for fi in (2.0, 3.0, 5.0):
    run_by_id("base_30k", tag=f"x{fi}", extra=f"--lambda_dssim {fi}")
```

Each tagged run writes to its own model dir, so `experiments.metrics` can score
them independently — pass the same `tag`:

```python
from experiments.metrics import extract_run
extract_run("base_30k", tag="x3.0")
```
```bash
python -m experiments.metrics --run-id base_30k --tag x3.0
```

## How it works

```
experiments/methods.py   Method registry (id -> repo, install, common args)
experiments/runs.py      Run manifest (run id -> method, iterations, extra args)
experiments/runner.py    prepare() + build_train_cmd() + run_method() + CLI
eval_worldsplat.py       on-the-fly renderer -> global + boundary/interior metrics
```

- **local methods** (`repo=None`, e.g. `worldsplat_prune`) run in *this* checkout.
- **external methods** are cloned into `<workdir_root>/<dir_name>`
  (default `/content/<dir_name>`) and pinned to `Method.ref` if set.
- Only the training submodules (`diff-gaussian-rasterization`, `simple-knn`) are
  initialized — the huge `SIBR_viewers` submodule is never cloned.
- `common_args` encode each method's baseline invocation; `vanilla_3dgs` uses
  `--eval --disable_viewer --quiet` so its train/test split matches the local runs.

## Adding a new baseline

Append to `experiments/methods.py`:

```python
register_method(
    Method(
        id="mip_splatting",
        label="Mip-Splatting",
        repo="https://github.com/autonomousvision/mip-splatting",
        ref="<tag-or-sha>",                 # pin for reproducibility
        dir_name="mip-splatting",
        install=_rasterizer_install(),      # adjust if its install differs
        common_args=("--eval", "--disable_viewer", "--quiet"),
    )
)
```

`install` and `common_args` are usually the only method-specific parts.

## Metric extraction

`experiments/metrics.py` turns a trained model into a row of the results sheet:
mean **PSNR / SSIM / LPIPS** per split (train & test), the **boundary-restricted**
and **interior** test PSNR, and the final **Gaussian count**.

```bash
python -m experiments.metrics --run-id base_30k                  # one run
python -m experiments.metrics --run-id base_30k --tag x3.0       # a sweep point
python -m experiments.metrics --sanity                           # the short sanity runs
python -m experiments.metrics --all                              # every non-sanity run
python -m experiments.metrics --model-dir <path> --name "adhoc" --iteration 30000
```

Rows are appended to `$HOME_DIR/output/metrics.csv` (override with `--csv`, or
`--no-csv` to print only). Columns:

```
Name, Run ID, Iteration, Train PSNR, Train SSIM, Train LPIPS,
Test PSNR, Test SSIM, Test LPIPS, Test PSNR band, Test PSNR interior,
Gauss. count, Train time (min), Eval time (s)
```

Computation is delegated to `eval_worldsplat.py` (with `--json_out`), which
renders each split on the fly. **The same metric code path scores every method**,
so pruned/upstream PLYs alike are directly comparable. The extractor ensures the
local rasterizer is installed first (same rasterizer-swap rule as training).

The **band/interior** split is dataset-driven: it comes from the class labels
under `<source_path>/semantic_class` (or a SAM2/instance dir), through the same
`worldsplat_prune` adapter. If they are absent, only the global metrics are
reported and the band/interior columns are blank.

## Evaluation notes

- Train and test PSNR/SSIM/LPIPS use the repo's canonical formulas from
  `metrics.py`: MSE-based PSNR, `utils.loss_utils.ssim`, and `lpipsPyTorch`
  (`vgg`, inputs in `[0, 1]`).
- Keep the protocol identical across methods (same `--eval` split, same
  iterations, same code).
- The boundary band is a `--boundary_band 4` (px) dilation of the class-boundary
  mask by default; override with `--boundary_band`.

## Troubleshooting

### External rasterizer build fails

`pip install ./submodules/diff-gaussian-rasterization` for an external method can
fail. The runner prints the **last lines of the build log** before raising, so the
real cause is visible. Common causes:

- **CUDA 13 / newer libstdc++ dropped the transitive `<cstdint>`** — the runner
  patches this automatically before every rasterizer build (guarded + idempotent).
- **Missing `glm`** — the stock `setup.py` hard-includes `third_party/glm/`; the
  runner force-inits it with `git submodule update --init --recursive`.
- **Toolchain mismatch** — nvcc's CUDA must match `torch.version.cuda`.
- **Rasterizer commit incompatible with a very new CUDA/torch** — set
  `rasterizer_ref="3dgs_accel"` on the `vanilla_3dgs` entry in
  `experiments/methods.py`. Re-run with `force=True` (CLI `--force`) to rebuild.

## Reproducibility

Pin `Method.ref` (a tag or commit SHA) once you settle on a baseline version.
`ref=None` means "whatever the default branch is at clone time", which is not
reproducible over time.
