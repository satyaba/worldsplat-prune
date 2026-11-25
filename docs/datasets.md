# Datasets — WorldSplat-Prune cross-dataset study

This page is the registry and preparation guide for the datasets used in the
WorldSplat-Prune experiments. The harness (`experiments/`) only ever sees a **dataset
directory** with a fixed layout; how the labels got there is adapter-specific.

## 1. Registry

| key | dataset path (under `$DATASET_ROOT`) | source | labels | adapter |
|---|---|---|---|---|
| `office0` | `office_0/Sequence_1` | Replica | GT class labels | `replica` |
| `garden`  | `mipnerf360/garden`   | Mip-NeRF 360 | SAM2 instance pseudo-labels | `sam2` |
| `bicycle` | `mipnerf360/bicycle`  | Mip-NeRF 360 | SAM2 instance pseudo-labels | `sam2` |

`$DATASET_ROOT` defaults to `$HOME_DIR/Dataset`. `garden`/`bicycle` are run by
pointing a baseline run at the scene (`run_by_id("base_30k", dataset="mipnerf360/garden")`,
or the CLI `--dataset`), so no manifest edit is needed to run them. `office_0`
remains the GT-segmentation reference scene; **no second Replica scene is
required** — the cross-dataset study runs on the two Mip-NeRF 360 scenes above.

## 2. Required on-disk layout

```
<DATASET_ROOT>/<dataset>/
  images/                     # RGB frames (Replica: rgb_<N>.png)
  sparse/0/                   # COLMAP model, PINHOLE / SIMPLE_PINHOLE camera
  semantic_class/             # OPTIONAL — GT class labels   (adapter: replica)
  semantic_instance/          # OPTIONAL — instance labels    (adapter: sam2)
```

`semantic_class` wins for Replica; `semantic_instance` is the SAM2 target. A
scene may have neither: WorldSplat-Prune runs then fail to build a store, and metrics fall
back to global PSNR/SSIM/LPIPS only (boundary/interior columns blank).

> **PINHOLE is mandatory.** `scene/dataset_readers.py` asserts a
> `PINHOLE`/`SIMPLE_PINHOLE` camera, and the rasterizer assumes an undistorted
> pinhole. SfM that leaves radial distortion must be undistorted (see §3).

## 3. Preparing `garden` / `bicycle` (Mip-NeRF 360, no GT segmentation)

End-to-end via `Prepare_Datasets.ipynb`; the two scripts it calls are:

### 3.1 SfM with HLOC → 3DGS-ready COLMAP

```bash
python -m worldsplat_prune.prep.hloc_sfm \
    --images "$HOME_DIR/raw/mipnerf360/garden/images" \
    --out    "$DATASET_ROOT/mipnerf360/garden" \
    --feature superpoint_aachen --matcher superglue \
    --pairs exhaustive            # or `--pairs sequential --window_size 20`
```

HLOC runs features/matching/reconstruction (default `SIMPLE_RADIAL`); the script
then **undistorts** (COLMAP `image_undistorter`, exactly what the repo's
`convert.py` does) into `<out>/images` + `<out>/sparse/0` with a PINHOLE camera.
If your frames are already rectified, add `--camera_model PINHOLE --no_undistort`.

### 3.2 SAM2 instance pseudo-labels

SAM2 is **class-agnostic**, so ids are *instances*, not classes — see §4.

```bash
python -m worldsplat_prune.segment_sam2 \
    --dataset_dir "$DATASET_ROOT/mipnerf360/garden" \
    --sam2_ckpt   "$HOME_DIR/checkpoints/sam2.1_hiera_large.pt" \
    --sam2_cfg    configs/sam2.1/sam2.1_hiera_l.yaml \
    --min_area 100
```

Writes `<dataset>/semantic_instance/<stem>.png` (uint16 ids, `0 == void`). Masks
are composed greedily by predicted-IoU + stability (higher score wins overlaps),
tiny masks are dropped, and uncovered pixels are nearest-filled so they do not
masquerade as boundaries. More masks/frame → pass a smaller `--min_area`; too
slow → `--resize_long_side 1024`.

### 3.3 Verify alignment

```bash
python -m worldsplat_prune.verify_alignment \
    --dataset_dir "$DATASET_ROOT/mipnerf360/garden" --adapter sam2
```

Overlays segment boundaries (red) on the RGB frames under `worldsplat_align_check/`;
check that the edges trace real object/region boundaries before training.

### 3.4 Consistent instances (minimal fusion)

Raw SAM2 ids restart every frame, so the same object gets different ids per view.
The post-hoc vote pass transfers one instance id per Gaussian and needs a
**single id per 3D object across views**. `worldsplat_prune.prep.fuse_instances`
produces that, training-free,
from posed RGB + COLMAP (**no depth, no classifier**):

```bash
python -m worldsplat_prune.prep.fuse_instances \
    --dataset_dir "$DATASET_ROOT/mipnerf360/garden" \
    --labels_dir  semantic_instance          # raw SAM2 maps (input)
    --out_dir     semantic_instance_fused    # consistent maps (output)
    --min_shared_points 2 --min_views 2 [--color_sim 0.5]
```

Then point WorldSplat-Prune at the fused maps:

```bash
# train/eval with the fused labels
--label_adapter sam2 --label_dir semantic_instance_fused
```

Algorithm (a stripped MaskClustering / SAMPro3D): lift each mask to 3D by keeping
COLMAP points that project onto it → merge masks whose **3D point sets overlap**
enough (`--metric jaccard|overlap` with `--min_score`, plus an optional
`--color_sim` gate) → clusters seen in ≥ `--min_views` frames become global
instances, the rest fall to void (B never budgets id 0). Output is per-view
`semantic_instance_fused/<stem>.png` plus `point_labels.npy` (per-COLMAP-point ids)
and `fusion_summary.json`.

> **Association choice matters.** The default is **overlap** (`--metric jaccard`),
> *not* the old shared-point-count rule (`--metric shared`). The shared rule
> single-links through hub points and collapses a whole scene into one instance —
> watch for the `[fuse] WARNING: only one instance …` line. If a scene still
> over-merges, lower `--min_score` won't help; instead re-label with
> `segment_sam2 --no_fill_void` (don't tile the background) and/or raise
> `--min_score`. `fusion_summary.json` reports `instances`, `largest_instance_frac`
> and size percentiles — `instances == 1` or `largest_instance_frac > 0.6` means
> the labeling is degenerate and pruning numbers are not meaningful.

**Ingesting off-the-shelf methods.** MaskClustering/SAMPro3D emit *per-point-cloud*
labels. `worldsplat_prune.prep.scene_io.splat_point_labels_to_views(scene, labels, out_dir)`
rasterises those into the same per-view `semantic_instance/` interface, so the
4-way comparison (no-fusion / minimal fusion / MaskClustering / SAMPro3D) all land
in the identical WorldSplat-Prune input format.

### 3.5 Instance-segmentation metric comparison (only with a grounding dataset)

The Mip-NeRF 360 scenes have **no ground-truth instances**, so the fusion cannot
be ranked against GT there. When a scene *with* GT per-view instance maps is
available (e.g. the full Replica release), score the labelings against it — every
method is lifted to the *same* COLMAP points and the *same* GT, so it is
apples-to-apples:

```bash
python -m worldsplat_prune.prep.eval_instances --dataset_dir "$DATASET_ROOT/room_0/Sequence_1" \
    --gt semantic_instance_gt \
    --method fused=semantic_instance_fused \
    --method raw=semantic_instance:first \
    --method maskclustering=/path/to/mc_point_labels.npy
```

- The GT is built once by lifting per-view `semantic_instance` maps onto the
  COLMAP points (majority vote) — or pass a real 3D GT with `--gt_npy`. The scene
  must ship GT instance maps (the standard Replica release does; some trimmed
  copies, e.g. `office_0` here, do not).
- Method inputs are either a production `point_labels.npy` (our fusion,
  SAMPro3D, MaskClustering, ...) or a per-view dir (`:first` reproduces the raw
  first-wins case, `:majority` is the default).
- Reports **AP25 / AP50 / mAP**, matched **mIoU** (at AP50) and **coverage**;
  predictions are ranked by mask size (no confidence scores are available
  zero-shot). `--iou`, `--min_area` and `--json_out` are available.

**Without a grounding dataset** (the Mip-NeRF 360-only plan) the fusion is treated
as *infrastructure*, not a benchmarked contribution, and the study leans on the
GT-free signals: the **partition-movement diagnostic** (Jaccard/TV between the
semantic and geometry-only kept sets — if the partition barely moves the *set*,
the arm is inert regardless of score quality), instance **consistency** stats in
`fusion_summary.json` (instances, support histogram), and the downstream render
metrics (PSNR/SSIM/LPIPS, band/interior, Gaussian count) against the raw and
geometry-only controls.

**GT caveat.** Replica's instance labels are per *view*; lifting them onto the SfM
points is a proxy for true 3D GT. It is consistent across all methods and is the
standard substrate for zero-shot 3D-instance evaluation when only 2D GT exists,
but absolute numbers should be read as comparative, not as a leaderboard.

### 3.6 Per-class fusion → composite maps (prompt-driven front-end)

The prompt-driven front-end (§2.2) labels **per class**: one per-frame instance-map
directory per phrase (SAM2 box-prompted). Fusing each class independently keeps
association from ever crossing classes, and the per-class instance ids are then
composed into the canonical **composite** id `class_id * 10000 + instance_id`:

```python
from worldsplat_prune.prep.fuse_instances import fuse_instances_per_class

summary = fuse_instances_per_class(
    dataset, {"chair": "sem_chair", "table": "sem_table"},
    metric="jaccard", min_score=0.25, min_views=2)
```

Writes `<dataset>/semantic_composite/<stem>.npy` (int32 composite maps),
`composite_point_labels.npy`, and `semantic_instance_fused/class_<cid>/` per-class
fused maps. Point the harness at the composite maps:

```bash
--label_adapter composite        # alias: grounded_sam2
```

### 3.7 Prompt-driven front-end (Grounded-SAM-2)

For a scene whose labels come from the open-vocabulary front-end (§2.2) rather
than SAM2 AMG, seed a phrase list, then detect → box-prompt SAM2 → per-class fuse
→ compose, in one driver:

```bash
# 1. seed phrases from frame captions (optional; curate the file by hand)
python -m worldsplat_prune.frontend.caption \
    --dataset_dir "$DATASET_ROOT/mipnerf360/garden" --out phrases_seed.txt --n 16

# 2. detect -> SAM2 box-prompt -> per-class fusion -> composite maps
python -m worldsplat_prune.prep.build_semantic_maps \
    --dataset_dir "$DATASET_ROOT/mipnerf360/garden" \
    --phrases phrases.txt \
    --sam2_ckpt "$HOME_DIR/checkpoints/sam2.1_hiera_large.pt"
```

Stages are skippable (`--skip_detect` reuses `detections.json`, `--detect_only`
stops before SAM2), and the intermediates (`detections.json`,
`semantic_instance_<class>/`, `semantic_instance_fused/`) are kept for
inspection. Detector and masking have their own CLIs
(`worldsplat_prune.frontend.grounding`, `worldsplat_prune.frontend.sam2_box`).
**Validate the predicted classes against Replica GT before trusting any pruning
result** (risk §9):

```bash
python -m worldsplat_prune.prep.eval_instances --dataset_dir <replica_scene> \
    --class_gt semantic_class --level class \
    --method frontend=semantic_composite
```

`eval_instances` also takes `--level {composite,class,instance}` to score a
chosen slice of a composite map (instance AP on the instance half, class accuracy
against class GT).

## 4. Label-adapter contract (why instance-only is enough)

Every adapter produces the canonical per-view map `L_v : int32 [H,W]`, `0 == void`;
the rest of WorldSplat-Prune never knows the source. See `docs/worldsplat_prune_spec.md` §2.

- **Replica** (`replica`): GT **class** labels remapped to `1..K` (this copy has
  no `semantic_instance/`, so a segment degenerates to a class).
- **SAM2** (`sam2` / `instance_dir`): per-frame **instance** ids. Unlike Replica
  classes these are *not* consistent across views — which is fine for the
  boundary geometry (`dt`/normals) that `SemanticStore` derives from `L_v`, but
  cross-view instance identity (the vote pass, target protection) needs the fused
  maps from `worldsplat_prune.prep.fuse_instances`.
- **Composite** (`composite` / `grounded_sam2`): the front-end's per-object,
  class-tagged maps — `id = class_id * 10000 + instance_id` (§3.6). Stored as
  int32 `.npy` (composite ids exceed the 16-bit PNG range); the adapter also reads
  integer image maps.

Related flags: `--label_adapter {replica,sam2,instance_dir,composite,grounded_sam2}`,
`--label_dir <dir>` (override the label directory),
`--min_seg_area N` (drop segments smaller than N px).

## 5. Running

```bash
# one dataset family
python -m experiments.runs --run garden_b10_30k
python -m experiments.metrics --run-id garden_b10_30k

# replay an existing arm on another scene without editing the manifest
python -m experiments.runs --run base_30k --dataset room_0/Sequence_1
```

All arms share the `--eval` (llffhold=8) split and 30k iterations, so rows are
comparable across datasets. **Caveat:** seed variance is ±0.1–0.2 dB PSNR
(see `docs/worldsplat_prune_spec.md` §3) — repeat before claiming a per-dataset delta.

## 6. Caveats / risks

- **Label noise.** SAM2 masks are not GT; a single bad frame pollutes that
  view's band/interior metrics. Spot-check with §3.3, and consider
  `--min_seg_area` to suppress speckle.
- **Instance inconsistency across views.** Per-view ids are fine for class-level
  use, but cross-view instance identity needs the fused maps
  (`worldsplat_prune.prep.fuse_instances`).
- **Resolution.** Training auto-resizes to ≤1600 px width; labels are resized
  nearest-neighbour to the camera resolution in `SemanticStore`, so no manual
  label resize is needed.
