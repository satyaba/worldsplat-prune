# WorldSplat-Prune — Design Spec (DRAFT)

### Post-reconstruction, semantic-aware Gaussian pruning for budget-controllable Level-of-Detail in urban digital twins

> **Project:** WorldSplat-Prune (Python package `worldsplat_prune`, renamed from `sgdc/`)
> **Status:** operator design complete (D1–D18); domain framing extended for
> **urban digital twins (UDTs)** (D19–D24); implementation pending.**
> This document captures the agreed design in full. The operator (D1–D18) is
> domain-agnostic and unchanged; the **domain framing moved from embodied AI to
> urban LoD** — the objective (§1), label source (§2.3), evaluation protocol
> (§6), risks (§8) and decisions (§7) are updated accordingly. The outstanding
> actions are the migration/cleanup (§9), the TUM2TWIN **reference-geometry
> adapter** (§2.3, §9.3), and the **UDT task metrics** (§6). The pruning operator
> (§4–5) is implemented (§9.3).
>
> **Branch:** `main` (renamed for the new project)
> **Intended fate:** this branch becomes a new project that **reuses** the
> semantic-labelling, dataset-prep and evaluation machinery already here, and —
> once the pipeline is finalized — **strips the unrelated SGDC density-control
> code** and replaces the SGDC design docs. See §9.

---

## 1. Objective

Given an **already trained** 3DGS model (a PLY; we do **not** re-run
reconstruction from scratch), remove Gaussians to meet a **target render
budget** while preserving the scene's **semantic structure**, so the reduced
asset is a usable, budget-controllable **Level-of-Detail (LoD)** product for an
**urban digital twin (UDT)** — where downstream tasks (LoD3 building
reconstruction, point-cloud semantic segmentation, roof/solar-potential
analysis) consume the asset for *semantic* and *geometric* reasoning, not just
for a human viewer.

The target domain is **urban scenes** — captured terrestrial, mobile and aerial,
and very large (tens of millions of Gaussians). The benchmark is **TUM2TWIN**
(§6): a georeferenced, multitemporal, multimodal UDT dataset with LoD1/2/3
building models, terrestrial/mobile/aerial laser scanning, imagery and ALS
semantic classes. The operator is **post-reconstruction and training-free**: it
operates on any frozen PLY produced by any pipeline — precisely the regime the
urban-LoD literature has *not* addressed (§1.1).

The claim to test is narrow and falsifiable:

> A semantic-aware pruning operator beats a **geometry-only** render-budget cap
> and a **size-matched random** control at the same Gaussian count — on
> reconstruction quality overall and on the **semantic-boundary band**
> specifically — where the band is the set of **urban class boundaries**
> (building/sky, building/vegetation, road/sidewalk, …), and the payoff is a
> **downstream UDT task**, not a viewer.

If it does not, the honest result is negative.

### 1.1 Framing: training-free, semantics-faithful LoD for urban digital twins

Urban 3DGS research has concentrated on **in-training** solutions — divide-and-
conquer training and progressive LoD (CityGaussian, VastGaussian, Hier-3DGS,
Octree-GS, Scaffold-GS, GoDe/CLoD-GS), anchor pruning (Urban-GS), and
contribution/sensitivity compression (MaskGaussian, ParetoGS). Semantic
*density control during training* also exists. Almost none of it reduces an
**already-built** asset, with a **semantic** prior, using the twin's own
**reference geometry**.

The object of study is therefore reframed:

> **Given a frozen urban 3DGS digital twin, produce a family of
> reduced-complexity assets that preserve the class boundaries, rooflines and
> thin structures that UDT downstream tasks depend on — *without retraining* —
> where geometry-only pruning degrades exactly those structures.**

The domain's own pain points are the motivation: the ISPRS 2025 evaluation of
3DGS for urban reconstruction reports that large urban scenes require tens of
millions of Gaussians, that occluded/boundary regions give unstable geometry and
high metric variance, and that deployment demands compression/LoD. This operator
targets that regime directly (§6).

### 1.2 Contributions to claim (pick 3–4; do not claim all)

| # | Contribution | Differentiator |
|---|---|---|
| **C1** | **Training-free, semantic-boundary-preserving pruning operator** (band protect + bounded purity confidence), with **class-conditioned δ** and a **geometry-edge term** (both implemented) | post-hoc ≠ in-training density control; semantic ≠ pure photometric/contribution pruning |
| **C2** | **Reference-geometry-anchored labeling** — the twin's **LoD models / LiDAR / ALS semantics** as the 3D semantic prior, not noisy per-view 2D maps (implemented: `tum2twin`/`reference` adapter + `label_from_reference` driver) | uses DT reference data absent from other pruning work; removes the 2D-label-noise confound |
| **C3** | **Budget-controllable "semantic LoD" family** — one frozen asset → discrete budgets (0.25/0.5/0.75) for streaming, with better boundary retention than baselines (export artifact implemented; class-conditioned budgets pending) | LoD is *semantically* defined in DTs (LoD1→3 = building detail); the band+class protection maps onto it |
| **C4** | **Task-aligned evaluation** — LoD-edge preservation, semantic-segmentation consistency, roof/solar-area error, LoD3 reconstruction, all at matched budget | replaces the old "no task metric" gap (§8) with the domain's own downstream tasks |
| **C5** | **Temporal consistency across UDT updates** — re-pruning stays consistent across multitemporal captures | only possible because the benchmark is multitemporal; hard to copy |
| **C6** | **Asset/landmark protection** (mode (b)) as a UDT control — preserve a chosen building/class/instance | application-motivated, cheap, evaluation-friendly |

The sharpest defensible set is **C1 + C2 + C4**; **C5** (temporal) is the hardest
to copy. See §8 for the novelty risks that must be stated explicitly.

### 1.3 Why this replaces the SGDC density-control line

The prior in-training study (`docs/sgdc_spec.md`, `docs/sgdc_report.md`) found
that **segmentation guidance was inert during training**: every semantic
mechanism (A, A.1, C, and B's per-segment split) failed its control, and the one
robust effect — ~11% fewer Gaussians at equal PSNR — came from a plain capacity
cap, not segmentation. The working hypothesis was that density control's target
(PSNR) and its signal (view-space photometric gradient) are the **same
modality**, so a mask adds little.

Post-reconstruction there is **no gradient at all**. The only signals are
geometry (opacity, scale) and semantics (labels). That is the regime where a
semantic prior may plausibly do work the geometry cannot — and it is why this is
a new research direction, not a continuation of SGDC.

---

## 2. Scope and reuse

| Item | Value |
|---|---|
| **Domain** | **urban digital twins** (TUM2TWIN, §6) — terrestrial/mobile/aerial, multitemporal, very large |
| Input | a trained model dir (`point_cloud/iteration_<N>/point_cloud.ply`) |
| Trains anything | **No** |
| Reuses | `sgdc.datasets` label adapters, `sgdc.semantic_store.SemanticStore`, `sgdc.budget` allocator, `eval_sgdc.py` metric path |
| New | a post-hoc **class-vote pass** (§3), a **candidate/selection** operator (§4–5), and a **reference-geometry label adapter** (§2.3) |
| Evaluation | the same `eval_sgdc.py` / `experiments.metrics` code path as every other row, extended with **UDT task metrics** (§6) |

Gaussians are never labelled from a 3D-fused point cloud. The 3D instance fusion
(`sgdc/prep/fuse_instances.py`) operates on **COLMAP points** and exists only to
make per-view SAM2 ids consistent; semantics reach Gaussians exclusively through
per-view 2D maps re-projected onto the (final) Gaussian centres.

### 2.1 Label-source front-end

The operator consumes `L_v`/`seg` and is agnostic to how they were produced. The
codebase produces only two label sources today and neither is an RGB→class
classifier:

- **SAM2** (`sgdc/segment_sam2.py`, adapter `sam2`/`instance_dir`) — class-agnostic
  instance ids; made cross-view consistent by `sgdc/prep/fuse_instances.py`.
- **Replica GT** (adapter `replica`) — ground-truth **class** labels read from
  disk, not generated from RGB.

For the **urban framing the primary route is different**: the twin's own
**reference geometry** — LoD building models, laser-scanning point clouds and
ALS semantic classes — supplies a 3D semantic prior directly, with no RGB
classifier and no per-view 2D pseudo-labels. It is specified in **§2.3**, and it
**removes the "things-only" coverage weakness** (§8) that afflicts image-based
detection on stuff-heavy urban scenes. The prompt-driven 2D front-end (§2.2) is
retained only as the **fallback for scenes without reference data**, and as the
`2D vs 3D reference labels` ablation.

### 2.2 Class front-end — prompt-driven, multi-class (fallback / ablation)

> **Status under the urban framing.** This route is **no longer the primary
> label source**; §2.3 (reference geometry) is. Keep it as the **fallback** for
> scenes without LoD/LiDAR, and as the `2D vs 3D reference labels` ablation.
> D14–D17 remain valid *for this route only*.

**Decision [D14]:** the class map is produced **prompt-driven / open-vocabulary** —
**no fixed taxonomy**; classes are text phrases. This is the "multi-class, no
taxonomy" route.

```
phrase list ─► GroundingDINO (open-vocab detector) ─► boxes
            └─► SAM2 (box-prompted) ─► masks ─► per-frame instance maps
                    └─► fuse_instances (per class) ─► consistent instance ids
                            └─► composite (class, instance) id map  ==  L_v
```

- **Granularity [D17]: per-object, class-tagged.** SAM2 masks are the *objects*
  (instances); GroundingDINO supplies the *class* phrase. The map is composite
  `(class_id, instance_id)` — segmentation is **per-object**, labels are
  **per-class**. Class-level operations group instances by class; instance-level
  operations (mode (b), §4.3) use the instance id. Per-object granularity is why
  cross-view **fusion** is required (class ids are consistent by construction;
  instance ids are not).
- **Phrase source [D15]:** an **MLLM auto-captioner seeds** the phrase list
  (offline — a small VLM or an API, run over a frame sample, then de-duplicated)
  and a **human curates** the final list ("both"). No hand-written taxonomy, but
  the vocabulary is checked before use.
- **Detector:** GroundingDINO (v1.5/1.6 an option) — text→boxes, free phrases.
- **Masking:** SAM2 prompted by the detector's boxes. NB `sgdc/segment_sam2.py`
  currently uses SAM2's *automatic* generator (AMG); this front-end adds the
  **box-prompted** mode (same weights).
- **Per-class fusion — always [D16]:** masks are written to the per-frame
  instance-map format (`masks_to_instance_map`), then `sgdc/prep/fuse_instances.py`
  runs **per class phrase**, so association never crosses classes → consistent
  instance ids. Class ids are consistent by construction; fusion supplies
  instance identity (also feeding mode (b), §4.3).
- **No video propagation** — COLMAP-based fusion instead (view-independent, works
  on unordered captures).
- **Void:** undetected pixels and unfused / `min_views`-dropped masks → `0`.
- **Edges:** SAM2 crisp masks → the δ band keeps the boundary quality the pruning
  design relies on.
- **Coverage caveat:** GroundingDINO detects **things**, so **stuff** classes
  (wall/floor/sky) appear only if explicitly prompted, and detection there is
  weak — unlike a mask-classifier route, expect sparse, stuff-poor labels.
- **Validation:** score predicted classes against the **reference-geometry
  classes** (§2.3) — or Replica GT where available — before trusting any pruning
  result.

Reused vs new: reuses `sgdc/segment_sam2.py` (SAM2 weights + composition),
`masks_to_instance_map`, `sgdc/prep/fuse_instances.py` and `scene_io`; **new** =
a GroundingDINO wrapper, an SAM2 box-prompted labeler, and a captioner→phrase-list
tool.

### 2.3 Reference-geometry front-end — urban LoD/LiDAR (chosen route)

**Decision [D19]:** in urban scenes the per-view label map `L_v` is derived from
the twin's **reference geometry**, not from RGB:

```
LoD1/2/3 building models ─┐
ALS / LiDAR point clouds  ─┼─► 3D semantic prior ──► label each frozen Gaussian
ALS 8-class semantics     ─┘        (class + instance)         │
                                                               ▼
                       per-view L_v by projecting labelled Gaussians into each
                       camera  ==  the canonical adapter contract (§4)
```

- **3D-first labelling.** Each Gaussian centre inherits a class from the nearest
  reference surface (LoD face / LiDAR point) within a distance *and*
  normal-agreement tolerance; the instance id comes from the LoD building/object
  id. Because the prior is 3D, ids are **cross-view consistent by construction** —
  the reference route does **not** need `fuse_instances`.
- **Classes = the ALS taxonomy** (unclassified, ground, building, water,
  object/vegetation, bridge, …) plus LoD building parts (roof/facade). This is a
  **fixed** taxonomy — deliberately: it is the DT's own standard. §2.2's [D14]
  "no taxonomy" applies to the *open-vocab route only*; the reference route is
  explicitly taxonomy-bound.
- **Per-view `L_v`** is written by projecting the labelled Gaussians (or by
  rasterising the labelled reference geometry) into each camera, so the rest of
  the pipeline — `SemanticStore`, the vote (§3), the operator (§4–5) — is
  unchanged.
- **Void / coverage.** Unmatched Gaussians (sky, unexplored space, thin
  structures the reference misses) → `0`; the bounded-confidence floor (§5) keeps
  them competing on geometry, so **void ≠ prunable** (§8 risk 2).
- **Validation.** Score reference-derived classes against held-out ALS labels and
  (on scenes that have them) against 2D semantic maps. The LoD models also serve
  as the **geometric GT** for the §6 task metrics.

**New components:** `worldsplat_prune/datasets/tum2twin.py` (adapter ids
`tum2twin` / `reference`) and a labelling driver `prep/label_from_reference.py`.
Reuses `semantic_store`, `scene_io` and the adapter contract; **not** the 2D
front-end. This is contribution **C2** (§1.2).

**Implemented (route (a)).** The driver rasterises labelled reference points into
per-view composite maps (`rasterize_composite`: project + z-buffer nearest-surface
wins + optional dilation) and writes `<DATASET_DIR>/semantic_reference/<stem>.npy`
+ `reference_meta.json` (taxonomy, instance source, consistency). The adapter
consumes them unchanged; a held-out ALS class map can be scored via
`--validate` (`confusion_summary`). Route (b) `assign_reference_labels_3d` assigns
labels to 3D points directly (nearest surface within a tolerance; unmatched →
void). Cores are numpy/cv2 and CPU-tested (`tests/test_tum2twin.py`); the
`--label_adapter reference` path is exercised end-to-end there against a synthetic
COLMAP scene.

---

## 3. The post-hoc class-vote pass

Training built a per-Gaussian semantic signature **online**, inside the
densification loop, and **never persisted it**:

- `GaussianModel.add_semantic_stats` (`scene/gaussian_model.py:632`), called from
  `train.py:208`, projects only the frame's *visible* Gaussians
  (`visibility_filter = radii > 0` from `render()`), reads `L_v`, and updates
  running buffers;
- the buffers (`seg_top1`, `seg_top1_cnt`, `seg_tot_cnt`, `seg_void_cnt`,
  `seg_dt_sum`, `seg_norm_sum`, `seg_err_sum`) are allocated only in
  `training_setup` (`:213-220`), are sliced/replicated on prune/clone/split, and
  are **absent on a loaded PLY** (`__init__` leaves them `torch.empty(0)`);
- `save_ply` (`:264`) writes only geometry + SH, so even an SGDC-trained model
  loses its vote on save.

Therefore, on a model we load and do not retrain, the vote must be **recomputed
post hoc** in a single deterministic pass. Call it `vote_classes_over_views`.

### 3.1 Inputs

- `gaussians` — loaded `GaussianModel` (final `xyz`, `scaling`, `rotation`,
  `opacity`);
- `cameras` — an **ordered** view list;
- `store` — `SemanticStore` bound to the dataset's label adapter;
- `visibility` — `render` | `project` (see §3.2);
- `vote` — `first` | `majority` (see §3.3);
- `dt_norm` — pixel normalizer for `d_i` (matches SGDC's `sgdc_dt_norm`);
- `min_views`, `dense_hist`, `chunk`.

### 3.2 Visibility: `render` vs `project`

The member set for the vote — the crux the old online path got for free.

| | `render` | `project` |
|---|---|---|
| Mechanism | CUDA rasterizer forward pass; returns `radii` and `visibility_filter = (radii > 0)` | one `[μ;1] @ full_proj_transform` + perspective divide; in-frame test |
| "Visible" | in frustum **and** footprint ≥ ~1 px | centre lands inside the image rectangle |
| Extra outputs | image, depth, per-Gaussian pixel radius | none |
| Occlusion | **not tested** | **not tested** |
| Cost / deps | full forward pass; CUDA + rasterizer + `pipe` | cheap; pure torch/numpy, CPU-ok |

Neither is a true z-buffer test: an in-frustum Gaussian behind a surface has
`radii > 0` and projects in-frame in both. `render` is the more faithful culling
(honours near/far planes and projected size); `project` is a cheap stand-in.

**Implementation note (Risk 1).** Because neither base test is depth-aware, a
depth-blind vote lets geometry *behind* an object inherit the object's label and
paints a ghost of it onto the background (observed on Mip-NeRF 360 `garden`
vases/tables). The post-hoc pass therefore adds an **optional occlusion test**
(`occlusion=True`, `depth_tol`): a sample only counts when the Gaussian is not
behind the visible surface at its pixel — the sheet is the rasteriser's
α-weighted inverse depth under `render`, or a per-pixel nearest-`z` buffer of the
centres under `project`. Occluded samples are dropped entirely (not counted as
void), so a Gaussian seen only through another object falls below `min_views`.
A second guard drops samples within `mask_margin` px of a label boundary (`dt`,
default 0): front-end silhouettes are a pixel or two generous, and those rim
pixels are where background geometry slips through the depth test. Both are
recorded in the bundle metadata (`n_occluded`, `n_masked`).

> **Urban framing.** The benchmark supplies **GT depth** and reference geometry
> (§2.3), so `occlusion=True` can be **validated** against reference depth rather
> than assumed — a rare check unavailable in image-only datasets.

Training used **both**: `render` for visibility, then `project` to find the
pixel (`add_semantic_stats`, `:653-660`). The post-hoc pass should follow the
same convention: **`render` by default [D6]**, `project` as an explicit
fast/preview mode. Record the choice in the output metadata.

### 3.3 Vote: `first` vs `majority`

The online rule is a **running first-label-wins top-1** (`:674-680`): the first
non-void label a Gaussian sees is frozen; later views only increment a match
counter. Post hoc we see *all* views, so a true majority (argmax over a class
histogram, ties → void) is affordable — but it is a *new* signal. Default
`first` **[D8]** keeps comparability with the SGDC-era semantics; `majority` is a
documented variant. The view **order** is now deterministic (training sampled
randomly) — define it (e.g. sorted by `image_name`) and record it, since
first-wins depends on order.

### 3.4 Output bundle (not model state)

A detached dict/dataclass, separate from the training buffers (which have a
different lifecycle and are empty on load):

| field | shape | meaning |
|---|---|---|
| `seg` | `long [N]` | assigned segment id (0 = void/unassigned) |
| `purity` `ρ` | `float [N]` | `seg_cnt / tot` (first) or `hist_max / tot` (majority) |
| `void_frac` `ν` | `float [N]` | `void / (labeled + void)` ∈ [0,1] |
| `boundary_dist` `d` | `float [N]` | `mean(dt)/dt_norm` ∈ [0,1] (0 = on a boundary) |
| `n_views` | `long [N]` | #views that sampled the Gaussian (confidence) |
| `seg_hist` | `float [N,K]` *optional* | class distribution (`dense_hist`) |
| `meta` | dict | `visibility`, `vote`, view order, `dt_norm`, `min_views`, counts |

Finalization: apply `min_views` (**default 3 [D7]**; below → `seg=0`, `ρ=0`);
never-seen (`tot==0`) → `seg=0`, `ρ=0`, `d=1.0` (conservative "interior"). Emit a
one-line summary
(`n_visible`, `n_labeled`, mean purity, mean void fraction) — the diagnostic used
to judge whether a scene's labelling is usable at all.

**DECIDED [D9] — the bundle is not extended.** No photometric-residual signal is
computed or stored. Geometry stays `α · Π s`; the only semantic quantities are
`seg`, `ρ`, `ν`, `d`, `n_views`. (Projected area is likewise not included.)

---

## 4. Candidate determination

**"Candidate" means the contested set only.** Protected and dead Gaussians are
settled by hard rules before any ranking. The pipeline, in order:

```
Input: loaded model (α, s, μ) + vote bundle (seg, ρ, ν, d, n_views)

(0) Gates — applied first, in this precedence:
    band_i      = d_i < δ                                    # absolute protect  [D1]
    target_i    = seg_i == s_target                          # instance protect  [D13]
    protected_i = band_i or target_i
    dead_i      = (α_i < ε) or (max(s_i) > 0.1·extent)       # hard-remove       [D5]
                  and not band_i        # band overrides dead; target does NOT (default)
    contested_i = not protected_i and not dead_i             # == the candidates

(1) Hard-remove all dead_i, unconditionally, outside the quota.
    N_rem = N - |dead|

(2) From contested only, delete K by score (higher score = keep):
    K = N_rem - N_target                                     # global budget    [D4]
    score_i = (α_i · Π s_i) · conf_i
    conf_i  = 1 - s·(1 - ρ_i^a)    # bounded; void & support off  [D2, D12]

(3) Record K_dead, K_contested, |protected|, final count.
```

### 4.1 The band (δ) — [D1] absolute protect

There are two scales of one idea:

- **2D band (the concept).** `semantic_utils.boundary_mask` marks pixels adjacent
  to a different label id (silhouettes); `distance_transform` gives `dt_v`, the
  Euclidean pixel distance to the nearest boundary. The band is the set of
  pixels within thickness **δ** of a boundary — the same object the eval metric
  calls the band (`eval_sgdc.py:_band_mask`, default `--boundary_band 4`).
- **Per-Gaussian band (the primitive).** `d_i = mean(dt)/dt_norm ∈ [0,1]`; the
  band is the threshold `d_i < δ` (with `dt_norm` default 100, a 4 px shell is
  `d_i < 0.04`).

**Why:** deleting Gaussians whose centre is on an edge is what makes two objects
visually merge or an object dissolve — the failure mode that ruins a scene as a
digital-twin LoD product (merged buildings, dissolved facades, lost thin
structures). Interiors are redundant; edges are structural, and labels are
noisiest there. So the band is where deletion is most damaging and most
ambiguous — leave it alone.

**DECIDED [D1] + [D11]:** absolute protect (`d_i < δ` ⇒ never a candidate), with
**δ = 4 px** default (matches `--boundary_band 4`, the codebase's established
band scale; with `dt_norm = 100` this is `d_i < 0.04`). Sweep {2, 4, 8} later.
Record δ in the metadata.

**DECIDED [D20] — urban band.** In the urban framing the band is the set of
**class boundaries** between ALS/LoD classes (building/sky, building/vegetation,
road/sidewalk, roof/facade, …) and is made **class-conditioned**: `δ` may differ
per class pair, and a **geometry-edge term** lets `d_i` also count depth/normal
discontinuities in the reference geometry, catching thin structures the class map
misses (wires, railings, scaffolding, tree canopies). The **class-conditioned δ
table is implemented**: `plan_pruning(band={class_id: δ})` (with `stride` /
`class_of` recovering the class half of a composite `seg`) applies `δ_i` per
Gaussian and records the table in the plan `meta`/`report`; the driver exposes
`--band_class class:delta` (repeatable) or a JSON path, and `experiments/runs.py`
passes it through. When unset the scalar `--band` (default 0.04) is used
unchanged, so the single [D1] `δ = 4 px` remains the default and existing runs
are bitwise unaffected. The **geometry-edge term is implemented**: from a
per-view depth map (`LabelAdapter.load_depth`, TUM2TWIN GT or rendered depth) a
depth-discontinuity mask + distance transform `dt_geom` gives a second boundary
distance ([`semantic_utils.geometry_distance_transform`]); the vote combines it as
`boundary_dist = min(d_class, d_geom) / dt_norm` **before** the operator (so
`prune.py` is untouched), and a thin depth ridge therefore protects a Gaussian
even where the class map is void. It is **opt-in** (`SemanticStore(edge_tol=…)` /
`--edge_tol`, default off) and additive-safe: with no depth it is inert.
Contributes to **C1**.

### 4.2 Hard-remove — [D5]

Baseline 3DGS deletes `α < ε` and `max(s) > 0.1·extent` **unconditionally** — it
is a prune rule, not a ranking (`densify_and_prune`, `:621-626`). We adopt the
same: the **dead** set is removed en masse, before and outside the quota.

Why keep a hard rule when the score already contains `α`: the score `α·Πs` can be
*high* for a large floater (high opacity, huge volume), so the hard rule catches
physical junk the score rates as important.

Post-hoc the rule is **scale-only**: `max_radii2D` is not stored in the PLY and
there is no optimizer state on load, so use `max(s) > 0.1·extent` (extent from the
scene/camera extent) plus `α < ε = 0.005` (the training call's value).

The dead set is **label-independent**, hence identical across every arm — a
constant offset that preserves comparability. Precedence: **band > dead >
contested** (a dead Gaussian in the band is protected). The **target instance**
(§4.3) is *budget-scope* by default — it excludes Gaussians from the budget but
does **not** override hard-remove, so dead Gaussians inside the target are still
removed.

**DECIDED [D11]:** `ε = 0.005` (the value in the training `densify_and_prune`
call) and `extent = scene.cameras_extent` (the same extent the baseline
scale-prune uses).

### 4.3 Target-instance protection — [D13] mode (b)

A *mode*: prune everywhere **except** a chosen object, which is kept intact. It
adds to the protect set: `target_i = (seg_i == s_target)`.

- **Granularity: instance.** `s_target` is a **consistent instance id** (the
  bundle's `seg`) — one object; protect every Gaussian whose top-1 vote is that
  instance. **Class-level protection** (all instances of a class) is a documented
  extension — see [D21] below.
- **Membership: blind.** `seg_i == s_target`, **no purity gate** — the band
  already guards the object's edges, so blind protection favours keeping the
  object whole over excluding a few mis-assigned floaters.
- **Override: budget-scope by default (ablatable).** The target keeps a Gaussian
  out of `contested` but does **not** override hard-remove — a dead Gaussian
  inside the target is still removed. The alternative, **full protect**
  (band-like, overrides dead), is an **ablation**: full protect would shelter
  always-eligible junk inside the object.
- **Requires consistent instance ids.** `seg` must be a *cross-view consistent*
  instance id, so SAM2 scenes need the fused maps (`sgdc.prep.fuse_instances`);
  raw per-frame SAM2 ids would break targeting. This **sidesteps the class
  front-end** (§ gap from the semantic-map discussion): instance targeting needs
  no RGB classifier — only a way to *choose* which instance id is the target
  (the deferred targeting/prompting step).
- **Reachability.** Let `P = |protected \ dead|` (band ∪ target, minus any that
  are hard-removed). Then `P ≤ N_target ≤ N − |dead|`; report when the target
  cannot be met (the protected object alone exceeds the budget).
- **Evaluation.** Add a **per-object** readout — PSNR on the instance's mask and
  its retained Gaussian count — since global/band means can hide a win for the
  object.

**DECIDED [D21] — class-level protection (urban).** Mode (b) also accepts a
**class** as the target (`s_target` ranges over a class id): protect every
Gaussian of that class. This is the natural UDT control — "keep all
buildings/facades intact while reducing ground and vegetation" — and maps
directly onto semantic LoD (C6). Instance targeting remains the default; class
targeting is an added mode, evaluated with a **per-class** readout alongside the
per-object one.

---

## 5. Score and selection

### 5.1 Score

The score ranks **contested** Gaussians only; the lowest-scoring `K` are deleted,
so **higher score = keep**. Protected and dead Gaussians never reach it. It
multiplies two reasons to keep:

```
score_i = (α_i · Π_j s_ij) · conf_i
```

**Geometry backbone `α_i · Π s_ij`** — volume-weighted opacity: "how much screen
energy does this carry?" This is the non-semantic control.

**Semantic confidence `conf_i ∈ [1−s, 1]`** — a *bounded* adjustment, so the
label signal tiebreaks among comparably-geometric Gaussians but never overrides
geometry or annihilates it:

```
conf_i = 1 − s · (1 − ρ_i^a)          s ∈ (0, 1)    # floor = 1 − s
```

- **`s` (semantic strength)** — the single sensitivity dial. `s = 0` ⇒
  geometry-only; `s = 1` ⇒ `conf = ρ` (no floor). **Default `s = 0.5`** — a
  Gaussian may lose at most half its keep-priority to the label signal.
- **`a` (curve)** — `a = 1` linear; `> 1` sharpens the penalty on low purity.
  **Default `a = 1`**.
- **`ρ` (purity / class agreement)** — the only semantic factor in the core.

**Why bounded, not a raw power.** `conf = ρ^a` (no floor) would give every
**never-labeled** Gaussian (`ρ = 0`: projects only into void, or below
`min_views`) `score = 0`, deleting the entire unlabeled background first — which
contradicts "void ≠ prunable". The `1 − s` floor keeps those Gaussians competing
on geometry.

**Void and support are off by default [D12]:**

- **`ν` (void) is not in the score.** Diagnostic only. Its old definition
  (`void / tot`, `tot` = labeled projections) is **not** a fraction (can exceed
  1), so `ν` is redefined as `void / (labeled + void)` before any use — and even
  then only in bounded form, since a raw `(1−ν)^b` hard-kills fully-void
  geometry.
- **`r(n_views)` (support)** — **ablatable, default off [D2]**. Low support ⇒
  weakly constrained, but conflates floaters with legitimately thin/occluded
  geometry, and is only meaningful under `render`.

**Ablation order:** `s` first (`{0.25, 0.5, 0.75}`), then `a`, then the void /
support factors.

**Multiplication, not division:** because we delete the *lowest*-scoring
Gaussians, confidence must *raise* the score. A bounded multiplier also lets
confidence **down-weight**, but never override, geometry.

**KEY SEMANTIC INTERACTION (band × purity):** low `ρ` is ambiguous — a floater
lands on different classes per view (remove), a genuine edge is where the
segmenter disagrees (protect). Disambiguation is by `d`: low `ρ` **far** from an
edge ⇒ artifact; low `ρ` **near** an edge ⇒ structure (already protected by the
band). This mirrors SGDC's Component-D band exclusion (`:376`), inverted.

### 5.2 Selection — [D4] global budget

Allocate the quota **scene-globally** (top-`K` by score). Decision [D4]: global
is the primary allocation; **per-segment** allocation is a documented ablation
only. Consequence: with global allocation the semantic content lives in (i) the
absolute band protect and (ii) the score's confidence term — not in *where the
quota goes*.

**The control.** With a global budget, `global` is no longer a distinct control.
The real control is **global + geometry-only score** (no band protect, no `ρ`
term). The control matrix:

| arm | score | band protect | allocation |
|---|---|---|---|
| control | geometry (`α·Πs`) | off | global |
| candidate | geometry × confidence | on | global |
| ablation | geometry × confidence | off | global |
| ablation | geometry × confidence | on | per-segment |
| structure bar | geometry | off | random |

**Mode (b) — target-instance protection (§4.3)** composes with *any* row above
(it only adds members to the protect set); its control is the same row with the
target **not** protected, judged by the per-object readout, not the scene mean.

### 5.3 Count contract

Final = `N_target` whenever `P ≤ N_target ≤ N − |dead|`, with
`P = |protected \ dead|`. Below the floor the protected object alone exceeds the
budget (unreachable); above the ceiling, dead removal overshoots
(`final = N − |dead| < N_target`) — **report** either. Report `K_dead` and
`K_contested` too: if `K_dead ≈ K`, the run is effectively geometry-only and
**no semantic delta can be attributed** — say so rather than after the fact.

---

## 6. Evaluation protocol

- **Dataset: TUM2TWIN** (urban, §2.3). The benchmark is multimodal
  (terrestrial/mobile/aerial images + LiDAR + LoD1/2/3 models + ALS semantics)
  and **multitemporal**. Reference geometry supplies labels (§2.3) and, through
  the LoD models, **geometric GT** for the task metrics below.
- **Tiling [D22].** TUM2TWIN's ~100,000 m² far exceeds a single 24 GB GPU, so a
  scene is processed as **tiles/subscenes** (BlockGaussian-style): prune per
  tile, reassemble, report per-tile and aggregate. Tiling is fixed across arms.
- **Models in:** trained PLYs from this repo (the base 3DGS trainer, **or** an
  external urban method's PLY — the operator is training-free, §1). The same
  `--eval` (llffhold=8) split is used for all rows.
- **View set:** **train-only [D10]** — the vote pass sees only
  `getTrainCameras()`, so no label or pruning decision leaks into the test split.
- **Arms:** the §5.2 control matrix at matched `keep_frac` ∈ {0.25, 0.5, 0.75};
  n ≥ 3 seeds (urban boundary metric variance is high — ISPRS 2025, §1.1).
- **LOD-family artifact (C3).** `prune_model.py --export_family --budgets
  0.25,0.5,0.75` runs load + §3 vote **once** and plans each budget on the shared
  operator, writing one standalone model dir per budget
  (`lod_<frac>/point_cloud/iteration_<n>/point_cloud.ply` + `cfg_args`) and a
  deterministic `lod_manifest.json` (per budget: matched count, full
  `PrunePlan.report()`, floor/ceiling reachability, kept-class histogram, δ
  table). Each `lod_<frac>/` is a valid model dir scored by the same metric path
  — no re-planning. `experiments.runs.build_lod_cmd` builds the command.
- **Metrics [D23] — two tiers:**
  - **Rendering (sanity, not headline):** `eval_worldsplat.py` → global
    PSNR/SSIM/LPIPS, **band** and **interior** PSNR (band = §4.1 urban class
    boundaries), Gaussian count, render cost.
  - **UDT task-aligned (headline, C4), at matched budget:**
    (i) **LoD-edge preservation** — distance error between the pruned asset's
    implied edges and the LoD model edges;
    (ii) **semantic-segmentation consistency** — IoU/mIoU of re-rendered labels
    vs the reference classes;
    (iii) **roof/solar-area error** — roof area implied by the pruned Gaussians
    vs the LoD roof area;
    (iv) **LoD3 reconstruction proxy** — facade/edge completeness.
    These are the signals the domain actually consumes.
- **Temporal consistency [D24] (C5).** On the multitemporal subsets, re-prune
  after a temporal update and measure **kept-set / semantic-LoD stability** across
  captures (per-budget Jaccard of kept classes). Reported separately.
- **Baselines:** geometry-only (control), random, and ≥ 1 urban compression/LoD
  method (a contribution/sensitivity pruner, or CityGaussian / Octree-GS where a
  retraining baseline is runnable). Position as **retraining vs our
  training-free** (§1.1, §8).
- **Diagnostics:** Jaccard/TV between the semantic and geometry-only kept sets —
  if the partition barely moves the *set*, the arm is inert regardless of score
  quality (the old pre-run diagnostic `sgdc/headroom` is removed — §9).

---

## 7. Decisions

### 7.1 Resolved

| id | decision | § |
|---|---|---|
| D1 | boundary band = **absolute protect** (`d_i < δ`) | §4.1 |
| D2 | `n_views` support term: **ablatable, default off** | §5.1 |
| D3 | band defined as a threshold on `d_i` | §4.1 |
| D4 | allocation: **global**; per-segment is an ablation only | §5.2 |
| D5 | dead Gaussians (`α<ε` or `max s>0.1·extent`): **hard-remove** | §4.2 |
| D6 | visibility default: **render** | §3.2 |
| D7 | `min_views` default: **3** | §3.4 |
| D8 | vote default: **first** | §3.3 |
| D9 | bundle **not** extended — no residual, no projected area | §3.4 |
| D10 | view set: **train-only** | §6 |
| D11 | hyperparameters: `δ=4px`, `ε=0.005`, `extent=scene.cameras_extent`, `dt_norm=100` | §4.1–4.2 |
| D12 | score: bounded confidence `conf = 1 − s·(1 − ρ^a)`, `a=1`, `s=0.5`; void (`ν`) & support (`n_views`) off; `ν` redefined as a true fraction | §5.1 |
| D13 | target protection (mode b): prune everywhere except a chosen **instance** (`seg == s_target`), blind membership, budget-scope by default (dead inside still removed); full-protect is an ablation | §4.3 |
| D14 | class front-end: **prompt-driven / open-vocabulary** (Grounded-SAM-2), **no taxonomy** | §2.2 |
| D15 | phrase source: **MLLM auto-caption seeds + human curation** ("both") | §2.2 |
| D16 | **per-class fusion always** via `fuse_instances`; no SAM2 video propagation | §2.2 |
| D17 | segmentation granularity: **per-object (instance), class-tagged** — composite `(class, instance)` map | §2.2 |
| D18 | project/package name: **WorldSplat-Prune** (Python package `worldsplat_prune`, from `sgdc/`) | §9.1 |
| D19 | **urban framing**: domain = UDTs (TUM2TWIN); primary label source = **reference geometry** (LoD/LiDAR/ALS), adapter `tum2twin`/`reference`; the 2D prompt-driven front-end becomes fallback/ablation | §1, §2.3 |
| D20 | band = **urban class boundaries**, **class-conditioned δ** (table implemented), plus a reference **geometry-edge term** for thin structures (implemented; opt-in) | §4.1 |
| D21 | target protection (mode b) gains **class-level** targeting (protect a whole class), for UDT semantic-LoD control | §4.3 |
| D22 | evaluation runs on **tiled/subscene** units (TUM2TWIN is ~100,000 m²), fixed across arms | §6 |
| D23 | headline metrics are **UDT task-aligned** (LoD-edge preservation, semantic-seg consistency, roof/solar-area, LoD3 proxy), not PSNR | §6 |
| D24 | **temporal consistency** across multitemporal captures is a first-class, separately reported metric | §6 |

### 7.2 Open

**None — structurally.** The operator is complete (D1–D18) and the urban framing
is settled (D19–D24); the only remaining action is review and promotion (§9).

**Label-source front-end:** the **primary** route is the reference-geometry
adapter (§2.3, [D19]) — LoD/LiDAR/ALS, new components `datasets/tum2twin.py` +
`prep/label_from_reference.py`. The prompt-driven Grounded-SAM-2 front-end
(§2.2) is retained as the **fallback/ablation**, not an unspecified gap.

**Still to *calibrate* (measurements, not open design questions):** the
class-conditioned `δ` table and the geometry-edge weight (§4.1), and the exact
UDT task-metric implementations (§6). All are TUM2TWIN-specific measurements.

### 7.3 Future option (not in the current design)

- **Promote per-segment allocation** from ablation to primary. Deliberately
  deferred [D4]; the allocator (`sgdc/budget.py: allocate()`) is **retained** (§9,
  ADAPT) for it. Revisit only if the semantic partition demonstrably moves the
  budget on the target scene (the old `headroom` diagnostic is removed;
  re-derive if needed).

---

## 8. Risks

1. **Centre-projection is footprint-blind** — mitigated by the band and by
   measuring how many pruned Gaussians were near a boundary.
2. **Void ≠ prunable** — `ν` is a soft prior only, never a hard delete.
3. **Low purity ≠ prunable** — disambiguate by `d` (band).
4. **SAM2 ids are per-frame** — the vote assumes cross-view consistency; use
   fused maps when it matters.
5. **Importance is heuristic** — a semantic win that does not beat
   geometry-only-global is not a semantic win.
6. **Task-metric fidelity (urban)** — the UDT task metrics themselves (§6) carry
   assumptions (edge extraction, roof-area estimation, segmentation model);
   validate each against the LoD/ALS reference before claiming a task win.
7. **Mode (b) needs consistent instance ids** — a target instance is only
   meaningful on fused (cross-view) SAM2 ids, and *choosing* the target is a
   deferred step; on raw per-frame ids the top-1 vote scatters and targeting
   breaks.
8. **2D front-end coverage is things-only** — GroundingDINO detects objects, not
   stuff, so walls/floor/sky stay void. Under the urban framing this is
   **mitigated by the reference-geometry route** (§2.3), which labels stuff
   directly; it affects only the 2D fallback/ablation arm.
9. **2D front-end labeling can be wrong** — captioner phrases may be hallucinated
   and per-class fusion leans on COLMAP point coverage (sparse objects → dropped
   to void). Keep **curate phrases + validate classes** for the fallback arm; the
   urban reference route sidesteps this but introduces its own alignment risk
   (risk 10).
10. **Reference-geometry alignment (urban, C2)** — projecting LoD/LiDAR onto
    Gaussian centres errs where geometry is thin, occluded or unmapped; the
    distance + normal-agreement tolerance is a knob. Validate class accuracy
    against held-out ALS labels (§2.3).
11. **Novelty vs in-training semantic density control** — we must **not** claim
    "semantic pruning" as new. The moat has to be stated explicitly: **post-hoc /
    training-free + reference-geometry labels + task-aligned eval** (§1.1). C1
    without those reads as incremental.
12. **Novelty vs urban LoD frameworks** (CityGaussian / Hier-3DGS / Octree-GS /
    CLoD-GS / Urban-GS) — do **not** claim to beat them on NVS. Claim
    *complementarity* (post-processing any pipeline's frozen asset) and *semantic
    fidelity per budget*; include one as a baseline.
13. **Scale / tiling** — TUM2TWIN is ~100,000 m²; pruning per tile can leave
    seams. Report per-tile and aggregate; check seam continuity (§6, [D22]).
14. **Boundary metric variance (urban)** — ISPRS 2025 documents high variance at
    occluded urban boundaries; a single seed is not evidence. n ≥ 3 and report
    spread (§6).

---

## 9. Migration & codebase cleanup

This branch becomes a **new project**. Two principles govern the end state:

1. **Reuse** machinery in this codebase that supports the research goal (label
   adapters, semantic store, dataset prep/fusion, the metric harness) rather than
   rebuilding it.
2. **Strip unrelated code** — when conceptualization is complete, no SGDC
   density-control code should remain that the new goal does not use.

**Status: IN PROGRESS (branch `main`).** The code-side inventory (KEEP / ADAPT /
REMOVE) is applied, the ADD front-end is implemented (pure parts unit-tested),
and the post-reconstruction pipeline is now implemented end to end: label lifting
(§3, `worldsplat_prune/vote.py`) and the pruning operator (§4–5,
`worldsplat_prune/prune.py` + the `prune_model.py` driver). Under the **urban
framing (D19–D24)** the prompt-driven front-end is **demoted to the
fallback/ablation arm** (§2.2) and a new **reference-geometry adapter** (§2.3)
becomes the primary label source. Three items remain: that adapter, the
pruning-ablation manifest (`experiments/runs.py`, whose flags/entry point
`prune_model.py` now defines), and the companion `docs/worldsplat_prune_report.md`
(depends on results). See §9.3. Paths below use the pre-migration `sgdc/...` names
for reference.

### 9.1 Code reuse & cleanup inventory

**KEEP — reused as-is (the semantic-labelling + evaluation machinery):**

- `sgdc/datasets/` (`base`, `replica`, `instance_dir`) — label adapters
- `sgdc/semantic_utils.py` — boundary mask, distance transform, normals, id-map IO
- `sgdc/semantic_store.py` — per-view label store
- `sgdc/prep/scene_io.py`, `sgdc/prep/hloc_sfm.py` — COLMAP IO + SfM prep
- `sgdc/verify_alignment.py` — label/image alignment check
- Core 3DGS: `scene/` (loader), `gaussian_renderer/`, `utils/`, `lpipsPyTorch/`,
  `submodules/`, `render.py`, `metrics.py`, `full_eval.py`, `convert.py`
- `train.py` (base training — used to *produce* models to prune)

**ADAPT — reused with changes:**

- `eval_sgdc.py` → the metric path (already scores vanilla PLYs); drop the SGDC
  naming, keep global + band/interior
- `sgdc/eval_boundary.py` → **kept** (evaluation module; consolidation with the
  metric path optional later)
- `sgdc/segment_sam2.py` → add **box-prompted** mode (§2.2; AMG-only today)
- `sgdc/prep/fuse_instances.py` → add **per-class** mode (§2.2)
- `sgdc/prep/eval_instances.py` → retarget to validate the front-end labelings
- `sgdc/budget.py` → keep `allocate()` (the per-segment allocator, §7.3); drop the
  Component-B error-budget logic (`per_gaussian_priority`, the `|grad|·e` term)
- `experiments/` (`runner`, `methods`, `metrics`) → keep; retarget the harness to
  run **ablations of the pruning method** (arms over `s`, `δ`, protect, ...)
- `experiments/runs.py` → replace the SGDC run specs with the **pruning-ablation
  manifest**
- `train.py`, `scene/gaussian_model.py`, `arguments/__init__.py` → remove SGDC
  hooks/flags, keep the base (see REMOVE)
- Notebooks (`Run_Experiments.ipynb`, `Prepare_Datasets.ipynb`) → retarget

**REMOVE — SGDC density-control, unrelated to the new goal:**

- `sgdc/budget_probe.py` — Component-B probe instrumentation
- `sgdc/headroom.py` — Component-B allocation diagnostic
- SGDC training components: `add_semantic_stats`, `compute_sgdc_thresholds`,
  `boundary_straddle`, `boundary_loss`, the `seg_*` buffers, and the
  `densify_and_prune` SGDC branches (Components A / A.1 / B / C)
- SGDC CLI flags in `arguments/__init__.py` (`sgdc_*`, `f_b`, `f_int`, `eta`,
  `lambda_bnd`, ...)
- SGDC run specs in `experiments/runs.py` + the `sgdc` method entry
- `docs/sgdc_spec.md`, `docs/sgdc_report.md`; the SGDC section of `results.md`

> `sgdc/budget.py: allocate()` is **retained** (see ADAPT) as the allocator the
> future per-segment option (§7.3) would need; only the error-budget logic is
> dropped.

**ADD — new components to build:**

- **(2D fallback/ablation, §2.2)** a **GroundingDINO wrapper** (text phrase →
  boxes); an **SAM2 box-prompted labeler** (extends `sgdc/segment_sam2.py`,
  AMG-only today — same weights, box-prompt mode); a **captioner → phrase-list
  tool** (small VLM or API, offline, de-duplicated); a **per-class fusion driver**
  over `sgdc/prep/fuse_instances.py`; and an **adapter** for the predicted
  composite `(class, instance)` maps.
- **(urban, primary, §2.3, C2)** a **reference-geometry label adapter**
  `datasets/tum2twin.py` (ids `tum2twin` / `reference`) consuming **LoD models,
  LiDAR/ALS point clouds and ALS classes**, plus a labelling driver
  `prep/label_from_reference.py`.
- **(urban, C4)** the **UDT task metrics** (§6, [D23]): LoD-edge preservation,
  semantic-segmentation consistency, roof/solar-area error, LoD3 proxy.

These are additions, not part of the reuse/cleanup of existing code. The urban
framing makes the reference-geometry adapter the **primary** label source (§2.3)
and the 2D front-end the **fallback/ablation** (§2.2).

**`tests/` and `scripts/` — follow their modules:**

- KEEP: `test_fusion.py`, `test_instance_labels.py`, `test_eval_instances.py`;
  `scripts/` (CUDA-submodule patch, build infra)
- REMOVE: `test_sgdc_budget.py`, `test_headroom.py`
- ADAPT: `test_experiments.py` → retarget to the pruning-ablation harness

**Naming — DECIDED: `WorldSplat-Prune`.** The project and package are renamed to
**WorldSplat-Prune** (Python import name `worldsplat_prune`; package dir `sgdc/` →
`worldsplat_prune/`), replacing the old "Segmentation-Guided Density Control"
name. All imports, CLI flags, docs and `experiments` paths update at migration
time.

### 9.2 Doc migration (replacement)

- `docs/sgdc_spec.md` → **replaced** by this document (promoted from DRAFT,
  renamed to the canonical `docs/worldsplat_prune_spec.md`);
- `docs/sgdc_report.md` → **replaced** by a companion
  `docs/worldsplat_prune_report.md`;
- `docs/datasets.md` and `experiments/README.md` → **adapted** (registry, prep
  and harness carry over; density-control sections removed). **Urban framing:
  the registry must add `tum2twin`/TUM2TWIN and the reference-geometry adapter,
  and re-role the Replica / Mip-NeRF 360 entries as fallback/ablation.**
- `README.md` → top-level description and links updated to the **urban-LoD**
  framing.

**Applied so far.** The SGDC design docs (`sgdc_spec.md`, `sgdc_report.md`) are
removed and `docs/worldsplat_prune_spec.md` is canonical; `docs/datasets.md`,
`experiments/README.md` and `README.md` are adapted (density-control sections
removed, front-end/fusion/composite added). The companion
`docs/worldsplat_prune_report.md` is still to be written once the pruning results
exist. Full per-item status: §9.3.

### 9.3 Implementation status

Phase legend: **Done** (implemented), **Done\*** (pure parts implemented and
unit-tested; the heavy model path needs `torch`/`transformers`/`sam2` + weights
and is *unverified* here), **Partial** (partially applied), **Blocked** (depends
on the §4–5 pruning operator), **N/A** (already satisfied).

| §9.1 item | phase | where / evidence |
|---|---|---|
| KEEP: label adapters, `semantic_utils`, `semantic_store`, `scene_io`, `hloc_sfm`, `verify_alignment`, core 3DGS, `train.py` | N/A | reused as-is |
| ADAPT: `eval_sgdc.py` → `eval_worldsplat.py` | Done | renamed, SGDC naming dropped; global + band/interior kept |
| ADAPT: `sgdc/eval_boundary.py` kept | Done | docstrings de-SGDC'd (commit `79388cb`) |
| ADAPT: `segment_sam2` box-prompted mode | Done\* | new `frontend/sam2_box.py` (`SAM2BoxLabeler`, `masks_to_class_maps`); pure composition tested (commit `45a2692`) |
| ADAPT: `fuse_instances` per-class mode | Done | `fuse_instances_per_class()` (commit `fac2821`); tested |
| ADAPT: `eval_instances` retarget | Done | composite-aware lifting + `--level {composite,class,instance}` + `--class_gt` validation (commit `fa094ad`); tested |
| ADAPT: `budget.py` keep `allocate()`, drop error-budget logic | Done | trimmed to the allocator only |
| ADAPT: `experiments/` retarget to pruning ablations | Done | harness/runner/metrics retained; `PRUNE_RUNS` grid + `run_prune*`/`extract_prune*` added |
| ADAPT: `experiments/runs.py` → pruning-ablation manifest | Done | `PRUNE_RUNS` over the control-matrix arms (semantic/geometry/random + noband/seg ablations), CLI `--list-prune/--prune/--all-prune` |
| ADAPT: `train.py`, `scene/gaussian_model.py`, `arguments/` de-SGDC | Done | no SGDC hooks/flags remain |
| ADAPT: notebooks retargeted | Done | repo URL/dir/branch + text updated (commit `79388cb`) |
| REMOVE: `budget_probe.py`, `headroom.py`, SGDC training components, CLI flags, run specs, SGDC docs | Done | removed (pre-existing + commit `79388cb`) |
| ADD: GroundingDINO wrapper | Done\* | `frontend/grounding.py` (HF backend); pure helpers tested (commit `fa094ad`) |
| ADD: SAM2 box-prompted labeler | Done\* | `frontend/sam2_box.py` |
| ADD: captioner → phrase-list tool | Done\* | `frontend/caption.py` (HF default, pluggable); pure helpers tested |
| ADD: per-class fusion driver | Done | `prep/build_semantic_maps.py` + `fuse_instances_per_class` |
| ADD: composite `(class, instance)` adapter | Done | `datasets/composite.py` (registered `composite`/`grounded_sam2`); tested (commit `fac2821`) |
| ADD: reference-geometry adapter `datasets/tum2twin.py` + `prep/label_from_reference.py` (§2.3, [D19]) | Done\* | primary urban label source (LoD/LiDAR/ALS → `L_v`): adapter ids `tum2twin`/`reference`; driver route (a) rasterises reference points to composite maps (`rasterize_composite`), route (b) `assign_reference_labels_3d`; `confusion_summary` validation; CPU-tested end-to-end (`tests/test_tum2twin.py`) |
| ADD: UDT task metrics — LoD-edge, semantic-seg, roof/solar, LoD3 (§6, [D23]) | Planned | headline urban eval beyond PSNR |
| Framing: domain pivot to **urban LoD** (§1, §2.3, §6; D19–D24) | Done | objective, framing/contributions, label source, protocol and risks all updated in this document |
| ADD: label lifting / class-vote pass (§3) | Done\* | `vote.py` (`VoteBundle`, `vote_classes_over_views`): render/project visibility, first/majority vote, depth occlusion + `mask_margin`; CPU-tested (`tests/test_vote.py`) |
| ADD: pruning operator (§4–5) | Done\* | `prune.py` (`plan_pruning`, `PrunePlan`) + `prune_model.py` driver; pure gates/score/selection CPU-tested (`tests/test_prune.py`), the model path is unverified here |
| ADD: class-conditioned δ ([D20], C1) | Done\* | `prune.py` `band={class_id: δ}` (+ `stride`/`class_of`, `parse_band_class`, `band_delta`, δ table in `meta`/`report`); driver `--band_class class:delta`/JSON; `experiments/runs.py` pass-through; CPU-tested (`tests/test_prune.py`, `tests/test_experiments.py`). Scalar default (0.04) unchanged; geometry-edge term still pending |
| ADD: LOD-family export (§6, C3) | Done\* | `export_lod.py` (`export_lod_family`, `plan_lod_family`, `family_manifest`, `kept_class_histogram`); CLI `--export_family`/`--budgets` reuses the driver's `load_and_vote` + subset/save path; `experiments.runs.build_lod_cmd`; CPU-tested (`tests/test_export_lod.py`, `tests/test_experiments.py`) |
| ADD: geometry-edge term ([D20], C1) | Done\* | `semantic_utils` `depth_edge_mask`/`geometry_distance_transform`/`distance_transform_from_mask`/`resize_depth`; `LabelAdapter.load_depth` hook; `SemanticStore(edge_tol=…)` returns `dt_geom`; `vote.py` combines `min(d_class, d_geom)` (meta `edge_term`/`n_edge_views`); driver `--edge_tol`. Opt-in, additive-safe; CPU-tested (`tests/test_semantic_utils.py`, `tests/test_vote.py`) |
| `tests/`: KEEP / REMOVE | Done | kept trio; SGDC tests gone |
| `tests/test_experiments.py` ADAPT | Done | retains the harness tests; prune-manifest tests added (grid, per-arm flags, matched budget) |
| Doc migration: spec replaced | Done | this document is canonical |
| Doc migration: `docs/worldsplat_prune_report.md` | Blocked | pending results |

**Test coverage:** `tests/test_composite.py`, `tests/test_vote.py` and
`tests/test_prune.py` (new) plus the extended `tests/test_frontend.py` /
`tests/test_eval_instances.py` cover the pure composition, encoding, lifting and
pruning paths; the full suite passes. The `Done\*` model paths are written
defensively but not exercised against model weights.

**Next work:** build the **UDT task metrics** (§6, [D23]); then run the
`PRUNE_RUNS` grid (arms × budgets × seeds) on a trained **TUM2TWIN tile**, collect
rows via `extract_prune_all`, and write `docs/worldsplat_prune_report.md`. The
reference-geometry adapter (§2.3, `datasets/tum2twin.py` +
`prep/label_from_reference.py`), label lifting (§3, `vote.py`), the operator
(§4–5, `prune.py`), the driver (`prune_model.py`), the LOD-family export
(`export_lod.py`, C3) and the ablation manifest (`experiments/runs.py`) are
implemented; the pure paths are CPU-tested (`tests/test_tum2twin.py`,
`tests/test_vote.py`, `tests/test_prune.py`, `tests/test_export_lod.py`,
`tests/test_semantic_utils.py`) and the CUDA `render` path is exercised only
against a GPU rasteriser.
