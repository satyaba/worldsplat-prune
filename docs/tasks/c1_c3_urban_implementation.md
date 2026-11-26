# C1–C3 urban implementation — agent handoff

Status: **open work**. Owner: any agent picking this up.
Context doc: [`../worldsplat_prune_spec.md`](../worldsplat_prune_spec.md) (urban framing, D19–D24).
Scope: make the pipeline genuinely support contributions **C1, C2, C3** (§1.2 of the spec).
Explicitly **out of scope**: C4 (UDT task metrics), C5 (temporal), C6 (landmark protection), and any retraining.

> Read §0, §3 and §4 before writing code. §3 is the honest current-state
> assessment; §4 lists the interfaces you must not break.

---

## 0. Read first

- The project is **WorldSplat-Prune**: a **post-reconstruction, training-free**
  semantic-aware Gaussian pruning operator for **urban digital twins**.
- The operator design (decisions **D1–D18**) is settled and considered
  domain-agnostic. The **urban framing** added **D19–D24** (see spec §1, §2.3, §4.1, §6, §7.1).
- The environment of record is **TUM2TWIN** (georeferenced, multitemporal,
  multimodal: LoD1/2/3 models, terrestrial/mobile/aerial LiDAR, ALS semantic
  classes, imagery). See spec §1, §6.
- **Do not retrain.** The whole point is that the operator acts on a frozen PLY.

---

## 1. Contribution definitions (spec §1.2)

| id | Contribution |
|---|---|
| **C1** | Training-free, **semantic-boundary-preserving** pruning operator (band protect + bounded purity confidence), with **class-conditioned δ**. |
| **C2** | **Reference-geometry-anchored labeling** — the twin's **LoD models / LiDAR / ALS semantics** as the 3D semantic prior, not per-view 2D maps. |
| **C3** | **Budget-controllable "semantic LoD" family** — one frozen asset → discrete budgets (0.25/0.5/0.75) for streaming, with better boundary retention than baselines. |

---

## 2. Target architecture (where new code lives)

```
frozen PLY
  │
  ├─(C2)─► reference-geometry labels ──► per-Gaussian VoteBundle (seg, purity, boundary_dist, …)
  │            (LoD / LiDAR / ALS)              │
  │                                             ▼
  └──────────────────────────────────(C1)─► plan_pruning ──► keep mask ──► pruned PLY
                                                     │
                                          (C3)─► export LOD family {0.25,0.5,0.75}
```

The operator (`worldsplat_prune/prune.py`) is the **single choke point**; C1
changes its gates/score, C2 feeds it, C3 calls it repeatedly.

---

## 3. Current-state assessment (verified against code)

### C1 — operator

| Component | Status | Evidence |
|---|---|---|
| Post-reconstruction / training-free | ✅ | `prune_model.py` (loads frozen PLY, no retrain); `worldsplat_prune/prune.py` is pure tensors |
| Band absolute protect `d_i < δ` | ✅ | `prune.py:209`, `DEFAULT_BAND = 0.04` (`prune.py:45`) |
| Bounded purity confidence `conf = 1 − σ(1 − ρ^a)` | ✅ | `prune.py:60-64` (`semantic_confidence`), applied `prune.py:240` |
| Control matrix (geometry-only / random / noband / per-segment) | ✅ | `plan_pruning` flags `use_band`/`use_purity`/`random_control`/`allocation`; driver `prune_model.py:161-169`; arms in `experiments/runs.py` (`PRUNE_ARMS`) |
| **Class-conditioned δ** | ✅ | `band` accepts `float` or `{class_id: δ}`; `stride`/`class_of`, `parse_band_class`, `band_delta`; driver `--band_class`; table in `meta`/`report` (T1) |
| **Reference geometry-edge term** (thin structures) | ✅ | `semantic_utils.geometry_distance_transform` + `depth_edge_mask`; `LabelAdapter.load_depth` hook; `SemanticStore(edge_tol=...)` → `dt_geom`; `vote.py` combines `min(d_class, d_geom)`; `--edge_tol` (T2) |

### C2 — reference-geometry labeling

| Component | Status | Evidence |
|---|---|---|
| Registered adapters | reference adapter registered | `datasets/__init__.py`: plus `tum2twin`/`reference` → `TUM2TWINLabelAdapter` |
| `datasets/tum2twin.py` | ✅ | present; consumes `<dataset>/semantic_reference/<stem>.npy` composite maps |
| `prep/label_from_reference.py` | ✅ | present; route (a) `rasterize_composite` + route (b) `assign_reference_labels_3d`, `confusion_summary` |
| Ingestion of LoD/LiDAR/ALS | ✅ | `.npz`/`.npy`/`.ply` reference clouds → composite `L_v` (T3) |
| Interface readiness | ✅ | the `LabelAdapter` contract (T4 below) plus `SemanticStore` → `vote_classes_over_views` means a new adapter plugs in **without touching the operator** |

C2 is the **largest** piece of work.

### C3 — budget-controllable LOD family

| Component | Status | Evidence |
|---|---|---|
| Global budget `keep_frac` / `n_target` | ✅ | `prune.py:227-228`; `prune_model.py:155-156` |
| Discrete budgets 0.25 / 0.5 / 0.75, matched counts | ✅ (as a grid, not a product) | `experiments/runs.py:114` `PRUNE_KEEP_FRACS = (0.25, 0.5, 0.75)` |
| **Packaged LOD-family artifact** | ✅ | `export_lod.py::export_lod_family` (T4): one load/vote → N `lod_<frac>/` PLYs + deterministic `lod_manifest.json`; CLI `--export_family` |
| **Class-conditioned budgets ("semantic LoD")** | ❌ | budget is one scalar; per-class allocation not wired (allocator exists: `worldsplat_prune/budget.py:allocate`) |

---

## 4. Interfaces you must respect (do not break)

### 4.1 Label adapter contract — `worldsplat_prune/datasets/base.py`

```
resolve(image_name) -> path | None      # resolve the label file for a training image
load(image_name)    -> int32 [H,W] | None   # canonical map, 0 == void
meta()              -> dict
```

The canonical id is a **composite `(class, instance)`**:
`id = class_id * STRIDE + instance_id`, `STRIDE = 10_000`
(`datasets/composite.py:39`; `encode`/`decode` helpers there). Class ids
`1..K`; instance ids `1..STRIDE-1`. Adapters with class-only data degenerate to
class-only (see `datasets/replica.py`).

Register new adapters in `datasets/__init__.py::_ADAPTERS`.

### 4.2 SemanticStore output — `worldsplat_prune/semantic_store.py:29`

```
get(image_name, H, W) -> {"label": int32 [H,W], "dt": float32 [H,W], "normal": float32 [H,W,2]} | None
```

`dt` is the class-boundary distance transform; `normal` is unused today.

### 4.3 Vote bundle — `worldsplat_prune/vote.py:70`

`VoteBundle(seg: long[N], purity: float[N], void_frac: float[N], boundary_dist: float[N], n_views: long[N], seg_hist?, hist_labels?, meta)`
All detached tensors; `seg = 0` is void.

### 4.4 Operator — `worldsplat_prune/prune.py::plan_pruning` (pure, device-agnostic)

Current signature (abridged):

```python
plan_pruning(alpha, scaling, *, boundary_dist, seg, purity, extent,
             keep_frac=0.5, n_target=None, band=DEFAULT_BAND, epsilon=..., strength=...,
             curve=..., use_band=True, use_purity=True, allocation="global",
             target=None, dead_extent_frac=..., random_control=False, seed=0) -> PrunePlan
```

Gate precedence is **band > dead > contested** (`prune.py:208-219`). Keep it.

### 4.5 Manifest / arms — `experiments/runs.py`

`PruneSpec(arm, keep_frac, strength, band, seed, …)`; `PRUNE_ARMS = ("semantic","geometry","random","noband","seg")`;
base model default `base_30k`; budgets `{0.25,0.5,0.75}`; strengths `{0.25,0.5,0.75}`; seeds `{0,1,2}`.
Arm→flags mapping: `runs.py:~180-205`.

### 4.6 Tests

Pattern: pure paths are **CPU-tested** under `tests/` (`tests/test_prune.py`,
`tests/test_vote.py`, `tests/test_composite.py`, `tests/test_experiments.py`).
New pure logic must ship with matching unit tests. Heavy model path
(`render` visibility) is exercised only against a GPU rasteriser.

---

## 5. Tasks

Each task: goal → files → approach → acceptance. Keep changes minimal and
backward-compatible; update the spec (§4.1/§2.3/§6) in the same PR.

---

### T1 — Class-conditioned δ (C1)

**Goal.** Allow the band threshold δ to differ per semantic class, so an urban
asset can protect e.g. building/facade boundaries more aggressively than ground.

**Files.** `worldsplat_prune/prune.py` (signature + gate), `prune_model.py`
(CLI), `experiments/runs.py` (pass-through), tests.

**Approach.**
- Let `band` accept **either** a `float` (current behaviour) **or** a
  `Mapping[int, float]` `{class_id: δ}`. Add a `stride: int = 10_000` /
  `class_of` hook so the operator can recover `class_id = seg // stride`.
- Compute `delta_i = table.get(class_id_i, default)` per Gaussian; the default
  is the scalar (D1 `δ = 4 px` → `0.04`). `band_mask = d_i < delta_i`.
- Add a CLI flag, e.g. `--band_class` accepting `class:delta` pairs
  (repeatable) or a JSON path; document the default = scalar (no behaviour
  change when unset).

**Acceptance.**
- With no mapping supplied, output is **bitwise identical** to today for all
  existing tests (regression guard).
- With a mapping, a synthetic test shows a Gaussian of a protected class with a
  slightly larger `d_i` is kept, and one of a non-protected class with the same
  `d_i` is contested.
- `meta`/`report` records the δ table used.
- New unit tests in `tests/test_prune.py`.

**Est.** ~0.5 day.

---

### T2 — Reference / geometry edge term (C1)

**Goal.** Catch **thin structures** the class map misses (wires, railings,
scaffolding, tree canopies) by adding a **geometry-edge** contribution to the
per-Gaussian `boundary_dist`.

**Files.** `worldsplat_prune/semantic_utils.py` (a geometry-edge helper),
`semantic_store.py` and/or `vote.py` (carry the second distance), `prune.py`
(combine), tests.

**Approach (pick one, document the choice):**
- **(a) Per-view depth discontinuity.** From GT depth (TUM2TWIN) or the rendered
  depth, build a geometry-boundary mask (depth/normal discontinuity) and a
  distance transform `dt_geom`; combine as
  `boundary_dist = min(dt_class, dt_geom) / dt_norm` before the operator (so
  `prune.py` is unchanged). Prefer negative depth discontinuities (occlusion
  edges) plus a gradient threshold; expose `--edge_tol`.
- **(b) 3D reference discontinuity.** Only if T3 route (b) exists — compute
  edge-ness from the reference geometry normals in 3D.

Prefer **(a)**: it reuses the existing per-view pipeline and needs no new 3D IO.

**Acceptance.**
- A synthetic frame with a thin depth ridge produces `dt_geom = 0` along it;
  Gaussians there become band-protected even where the class map is void.
- Combining is **additive-safe**: with no depth available, behaviour equals today.
- New unit test for the edge helper (`tests/test_vote.py` or a new
  `tests/test_semantic_utils.py`).

**Est.** ~0.5–1 day. **Note:** the class-conditioned part needs no reference data;
the edge term leans on GT depth from C2's data but can be built from rendered
depth first.

---

### T3 — Reference-geometry label adapter + driver (C2) — **the main build**

**Goal.** Produce the per-Gaussian semantic signature from the twin's **LoD /
LiDAR / ALS** reference geometry instead of RGB-derived 2D maps.

**New files.**
- `worldsplat_prune/datasets/tum2twin.py` — adapter ids **`tum2twin`** and
  **`reference`** (register both in `datasets/__init__.py`).
- `worldsplat_prune/prep/label_from_reference.py` — the labelling driver.
- `tests/test_tum2twin.py`.

**First decide the route (see §6). Recommended: route (a) first.**

**Route (a) — rasterize labelled reference geometry into per-view `L_v` (drop-in).**
- Inputs: posed cameras (`scene.getTrainCameras()`), reference geometry (LoD
  mesh `.glb`/`.obj`, LiDAR/ALS point clouds with class ids).
- For each camera: project/rasterize the reference surfaces; assign each pixel
  the reference **class** (ALS taxonomy) and **instance** (LoD building/object
  id); encode composite `id = class * STRIDE + instance`; write
  `<DATASET_DIR>/semantic_reference/<stem>.npy` (int32) exactly like
  `datasets/composite.py`.
- Implement `TUM2TWINLabelAdapter.load()` to read those maps → **zero changes**
  to `SemanticStore`, `vote.py`, `prune.py`.
- Reference-instance ids are **cross-view consistent by construction**; the
  driver should say so in `meta()` and skip `fuse_instances`.

**Route (b) — label Gaussians directly in 3D (more faithful, more code).**
- For each Gaussian centre, find nearest reference surface within a distance +
  normal-agreement tolerance; assign class/instance.
- Emit a `VoteBundle`-compatible object directly (bypasses `vote.py`): must fill
  `boundary_dist` consistently (distance to the reference class boundary, or via
  route (a)'s per-view `dt`), plus `n_views`/`purity` sensibly (e.g. purity from
  per-view agreement or 1.0 with a `meta` note).
- Keep `plan_pruning` untouched — feed it the same bundle fields.

**Acceptance (either route).**
- `python prune_model.py -m <model> --label_adapter reference --dry_run` produces
  a plan whose `report()["vote"]` shows `n_labeled > 0` and a sane
  `mean_purity`, on a TUM2TWIN scene with reference data.
- The adapter `meta()` reports the class taxonomy, instance source, and
  consistency guarantee.
- A pure CPU test covers encode/decode, nearest-surface assignment tolerance,
  and void behaviour for unmatched Gaussians (must be `0`, not protected).
- Reference classes can be validated against held-out ALS labels (report a
  confusion summary; see spec §2.3).

**Est.** 2–4 days depending on route and reference-data ingestion.

---

### T4 — LOD-family export (C3)

**Goal.** Turn "run the driver at each budget" into a **first-class artifact**:
one frozen asset → a manifest + N pruned PLYs.

**Files.** New `worldsplat_prune/export_lod.py` (or a `--export_family` path in
`prune_model.py`) + `experiments` hook + `tests/test_export_lod.py`.

**Approach.**
- `export_lod_family(model_dir, budgets=(0.25,0.5,0.75), …, out_dir)`:
  load once, compute the vote bundle once, call `plan_pruning` per budget,
  write `<out>/lod_<frac>/point_cloud/iteration_<n>/point_cloud.ply`, copy
  `cfg_args`, and write `<out>/lod_manifest.json` with, per budget: Gaussian
  count, `PrunePlan.report()`, class histogram of kept Gaussians, δ table, meta.
- Reuse the exact driver path (`prune_model.run`) so metrics stay comparable;
  do not fork the operator logic.

**Acceptance.**
- One invocation writes all budgets; `lod_manifest.json` is deterministic across
  runs (same seed).
- Each PLY is loadable by `eval_worldsplat.py` / `experiments.metrics`.
- Counts match `round(keep_frac · N)` subject to the floor/ceiling contract
  (`prune.py:231-232`), reported when unreachable.
- Unit test on a tiny synthetic model set (no CUDA) for manifest structure and
  matched counts.

**Est.** ~1 day.

---

### T5 — Class-conditioned budgets ("semantic LoD") (C3)

**Goal.** Optional: allow the *budget itself* to be class-aware (e.g. keep all
buildings, halve ground/vegetation), so the exported LODs are semantically
defined rather than uniform.

**Files.** `worldsplat_prune/budget.py` (`allocate` exists), `prune.py`
(`allocation="per_class"`), `prune_model.py`, tests.

**Approach.** Reuse the existing `allocate()` over per-class weights; delete the
lowest-scoring within each class quickly until the global budget is hit. Keep
`allocation="global"` the default (D4); per-class is an ablation/extension.

**Acceptance.** New allocation is opt-in; global path unchanged; unit test with
matched class quotas.

**Est.** ~1 day. **Optional** — only needed if the paper claims the *semantic*
LOD framing, not just budget-controllable LOD.

---

## 6. Decisions to make (open)

1. **T3 route (a) vs (b).** Recommend **(a)** first: it keeps the operator and
   vote untouched, reuses the composite encoding, and gives instant
   cross-view consistency. Move to (b) only if reference→2D rasterization loses
   too much (thin structures, occlusion).
2. **δ table source (T1).** Hand-set per class pair vs. derived from reference
   boundary statistics. Start hand-set with a documented default.
3. **Edge-term source (T2).** GT depth vs rendered depth vs 3D reference normals.
   Prefer GT depth when available, rendered depth as fallback.

Record each choice in the spec (§7.1 decisions) when made.

---

## 7. Suggested order

1. **T1** (cheap, self-contained, high value).
2. **T4** (cheap, makes C3 presentable, unblocks experiments structure).
3. **T2** (medium; can start from rendered depth before reference data lands).
4. **T3** (main build; route (a)).
5. **T5** (optional, only if claiming semantic LOD).

---

## 8. Definition of done

- C1: per-class δ + edge term implemented, opt-in, default = current behaviour,
  unit-tested; regression suite green.
- C2: `--label_adapter reference` runs end-to-end on a TUM2TWIN scene; adapter
  + driver + tests merged; spec §2.3 marked Done.
- C3: `export_lod_family` writes a deterministic manifest + PLYs at
  `{0.25,0.5,0.75}` loadable by the metric path; spec §6 updated.
- Spec `docs/worldsplat_prune_spec.md` §1.2, §2.3, §4.1, §6, §7.1, §9.3 updated
  to reflect the new state.

---

## 9. Out of scope / do not touch

- **No retraining** anywhere; the operator is post-reconstruction by design (D19/C1).
- Do **not** change the gate precedence (`band > dead > contested`, `prune.py:208-219`)
  or the bounded-confidence form (`prune.py:60-64`) — those are settled (D1, D12).
- **C4 (UDT task metrics), C5 (temporal), C6 (landmark protection)** are separate
  work items (spec §6). Do not implement them here.
- Keep `allocation="global"` the default (D4).

---

## 10. File map (quick reference)

| Concern | File |
|---|---|
| Operator (gates/score/selection) | `worldsplat_prune/prune.py` |
| Label lifting / vote | `worldsplat_prune/vote.py` |
| Per-view store + dt/normals | `worldsplat_prune/semantic_store.py`, `worldsplat_prune/semantic_utils.py` |
| Label adapters | `worldsplat_prune/datasets/{base,replica,instance_dir,composite}.py`, `datasets/__init__.py` |
| Per-segment allocator | `worldsplat_prune/budget.py` |
| Driver | `prune_model.py` |
| Pruning manifest/arms | `experiments/runs.py` |
| Metric path | `eval_worldsplat.py`, `experiments/metrics.py` |
| Tests | `tests/test_{prune,vote,composite,experiments}.py` |
