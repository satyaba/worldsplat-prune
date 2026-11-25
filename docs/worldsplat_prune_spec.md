# WorldSplat-Prune — Design Spec (DRAFT)

### Post-reconstruction, semantic-aware Gaussian pruning for real-time simulations in embodied AI

> **Project:** WorldSplat-Prune (Python package `worldsplat_prune`, renamed from `sgdc/`)
> **Status:** design complete (D1–D18); implementation pending.**
> This document captures the agreed design in full; no structural items remain
> open. The outstanding actions are executing the migration/cleanup (§9) and
> building the new front-end (§2.2) and the pruning operator (§4–5).
>
> **Branch:** `main` (renamed for the new project)
> **Intended fate:** this branch becomes a new project that **reuses** the
> semantic-labelling, dataset-prep and evaluation machinery already here, and —
> once the pipeline is finalized — **strips the unrelated SGDC density-control
> code** and replaces the SGDC design docs. See §9.

---

## 1. Objective

Given an **already trained** 3DGS model (a PLY; we do **not** re-run
reconstruction from scratch), remove Gaussians to meet a **real-time render
budget** while preserving the scene's **semantic structure**, so the pruned
asset is usable as a simulation substrate for **embodied AI** — where a robot
consumes the render for collision/occupancy reasoning and stable per-object
appearance, not just for a human viewer.

The claim to test is narrow and falsifiable:

> A semantic-aware pruning operator beats a **geometry-only** render-budget cap
> and a **size-matched random** control at the same Gaussian count — on
> reconstruction quality overall and on the semantic-boundary band specifically.

If it does not, the honest result is negative.

### 1.1 Why this replaces the SGDC density-control line

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
| Input | a trained model dir (`point_cloud/iteration_<N>/point_cloud.ply`) |
| Trains anything | **No** |
| Reuses | `sgdc.datasets` label adapters, `sgdc.semantic_store.SemanticStore`, `sgdc.budget` allocator, `eval_sgdc.py` metric path |
| New | a post-hoc **class-vote pass** (§3) and a **candidate/selection** operator (§4–5) |
| Evaluation | the same `eval_sgdc.py` / `experiments.metrics` code path as every other row |

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

"Multi-class binary classification from RGB" is therefore **not** supported
today; the route adopted to close the gap is specified in **§2.2**.

### 2.2 Class front-end — prompt-driven, multi-class (chosen route)

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
- **Validation:** score predicted classes against **Replica GT classes** before
  trusting any pruning result.

Reused vs new: reuses `sgdc/segment_sam2.py` (SAM2 weights + composition),
`masks_to_instance_map`, `sgdc/prep/fuse_instances.py` and `scene_io`; **new** =
a GroundingDINO wrapper, an SAM2 box-prompted labeler, and a captioner→phrase-list
tool.

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
simulation substrate. Interiors are redundant; edges are structural, and labels
are noisiest there. So the band is where deletion is most damaging and most
ambiguous — leave it alone.

**DECIDED [D1] + [D11]:** absolute protect (`d_i < δ` ⇒ never a candidate), with
**δ = 4 px** default (matches `--boundary_band 4`, the codebase's established
band scale; with `dt_norm = 100` this is `d_i < 0.04`). Sweep {2, 4, 8} later.
Record δ in the metadata.

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
  instance. Class-level protection (all instances of a class) is a possible
  extension, not the default.
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

- **Datasets:** `office_0/Sequence_1` (Replica GT class labels); later Mip-NeRF
  360 `garden`/`bicycle` (SAM2 instances, fused maps for the vote).
- **Models in:** trained PLYs from this repo; same `--eval` (llffhold=8) split
  for all rows.
- **View set:** **train-only [D10]** — the vote pass sees only
  `getTrainCameras()`, so no label or pruning decision leaks into a metric later
  scored on the test split.
- **Arms:** the control matrix above, at matched `keep_frac` ∈ {0.25, 0.5, 0.75};
  n ≥ 3 (seed variance is real).
- **Metrics:** `eval_sgdc.py` → global PSNR/SSIM/LPIPS, **band** and **interior**
  PSNR, Gaussian count. Render cost read off the count.
- **Diagnostics:** Jaccard/TV between the semantic and geometry-only kept sets —
  if the partition barely moves the *set*, the arm will be inert regardless of
  score quality (the analogous pre-run diagnostic during training was
  `sgdc/headroom`, removed — §9).

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

### 7.2 Open

**None — structurally.** The pipeline is complete (D1–D18); the only remaining
action is review and promotion (§9).

**Label-source front-end:** specified in §2.2 (prompt-driven Grounded-SAM-2, no
taxonomy, auto-caption + curate phrases, always per-class fusion). It is a **new
component to build**, not an unspecified gap.

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
6. **No embodied-AI task metric yet** — band/interior PSNR and per-segment
   coverage are proxies.
7. **Mode (b) needs consistent instance ids** — a target instance is only
   meaningful on fused (cross-view) SAM2 ids, and *choosing* the target is a
   deferred step; on raw per-frame ids the top-1 vote scatters and targeting
   breaks.
8. **Front-end coverage is things-only** — GroundingDINO detects objects, not
   stuff, so walls/floor/sky stay void unless explicitly prompted (and are weak
   there); expect sparse, stuff-poor labels from the chosen route (§2.2).
9. **Front-end labeling can be wrong** — captioner phrases may be hallucinated
   and per-class fusion leans on COLMAP point coverage (sparse objects → dropped
   to void); **curate phrases and validate classes against Replica GT** before
   trusting any pruning result.

---

## 9. Migration & codebase cleanup

This branch becomes a **new project**. Two principles govern the end state:

1. **Reuse** machinery in this codebase that supports the research goal (label
   adapters, semantic store, dataset prep/fusion, the metric harness) rather than
   rebuilding it.
2. **Strip unrelated code** — when conceptualization is complete, no SGDC
   density-control code should remain that the new goal does not use.

**Status: IN PROGRESS (branch `main`).** The code-side inventory (KEEP / ADAPT /
REMOVE) is applied, and the ADD front-end is implemented (pure parts
unit-tested). Two items remain **blocked, not started**: the pruning-ablation
manifest (`experiments/runs.py`, depends on the §4–5 operator) and the companion
`docs/worldsplat_prune_report.md` (depends on results). See the per-item table in
§9.3. Paths in the inventory below are written with the pre-migration `sgdc/...`
names for reference.

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

**ADD — new components to build (the class front-end, §2.2):**

- a **GroundingDINO wrapper** (text phrase → boxes);
- an **SAM2 box-prompted labeler** (extends `sgdc/segment_sam2.py`, which is
  AMG-only today — same weights, box-prompt mode);
- a **captioner → phrase-list tool** (small VLM or API, offline, de-duplicated);
- a **per-class fusion driver** over the existing `sgdc/prep/fuse_instances.py`;
- an **adapter** for the predicted composite `(class, instance)` maps (the
  front-end's `L_v` consumer, alongside `sgdc/datasets/`).

These are additions, not part of the reuse/cleanup of existing code.

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
  and harness carry over; density-control sections removed);
- `README.md` → top-level description and links updated.

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
| ADAPT: `experiments/` retarget to pruning ablations | Blocked | harness/runner/metrics retained; the pruning arms need the operator |
| ADAPT: `experiments/runs.py` → pruning-ablation manifest | Blocked | deferred by decision; flags/entry point undefined until §4–5 lands |
| ADAPT: `train.py`, `scene/gaussian_model.py`, `arguments/` de-SGDC | Done | no SGDC hooks/flags remain |
| ADAPT: notebooks retargeted | Done | repo URL/dir/branch + text updated (commit `79388cb`) |
| REMOVE: `budget_probe.py`, `headroom.py`, SGDC training components, CLI flags, run specs, SGDC docs | Done | removed (pre-existing + commit `79388cb`) |
| ADD: GroundingDINO wrapper | Done\* | `frontend/grounding.py` (HF backend); pure helpers tested (commit `fa094ad`) |
| ADD: SAM2 box-prompted labeler | Done\* | `frontend/sam2_box.py` |
| ADD: captioner → phrase-list tool | Done\* | `frontend/caption.py` (HF default, pluggable); pure helpers tested |
| ADD: per-class fusion driver | Done | `prep/build_semantic_maps.py` + `fuse_instances_per_class` |
| ADD: composite `(class, instance)` adapter | Done | `datasets/composite.py` (registered `composite`/`grounded_sam2`); tested (commit `fac2821`) |
| `tests/`: KEEP / REMOVE | Done | kept trio; SGDC tests gone |
| `tests/test_experiments.py` ADAPT | Partial | SGDC example flags swapped; pruning-harness retarget is **Blocked** |
| Doc migration: spec replaced | Done | this document is canonical |
| Doc migration: `docs/worldsplat_prune_report.md` | Blocked | pending results |

**Test coverage:** `tests/test_composite.py` (new) and the extended
`tests/test_frontend.py` / `tests/test_eval_instances.py` cover the pure
composition, encoding and evaluation paths; the full suite passes. The `Done\*`
model paths are written defensively but not exercised against model weights.

**Not covered by §9** (next work): the post-reconstruction pruning operator itself
(§4–5), which unblocks the pruning-ablation manifest and the report. The label
lifting that feeds it — the `vote_classes_over_views` pass (§3) — is now
implemented in `worldsplat_prune/vote.py` (`VoteBundle`, `vote_classes_over_views`),
CPU-tested in `tests/test_vote.py`; the CUDA `render` membership path is wired but
exercised only against a GPU rasteriser.
