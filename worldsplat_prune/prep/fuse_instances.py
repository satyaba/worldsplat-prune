"""Minimal training-free **3D instance fusion** from per-frame SAM2 masks.

The problem: SAM2 AMG emits class-agnostic masks whose ids restart every frame,
so the same object gets different ids in every view. The post-hoc vote pass needs
a **single id per 3D object across views**, so each Gaussian can carry one
consistent instance label. This module produces that by fusing
per-frame masks through the COLMAP point cloud -- *no depth, no classifier, no
per-scene training*.

Algorithm (deliberately a stripped MaskClustering / SAMPro3D):

1. **Lift** every mask to 3D by projecting the COLMAP points into its frame and
   keeping the points that fall on the mask -> a 3D point set per ``(frame, mask)``.
2. **Associate**: masks that share ``>= min_shared_points`` 3D points vote to be
   the same object (union-find). An optional mean-colour gate blocks merges
   across an object boundary (a boundary point is seen by both neighbours).
3. **Consolidate**: each connected component is one global instance; components
   seen in fewer than ``min_views`` frames are dropped to void (B never budgets
   id 0, so this is a free confidence gate).
4. **Relabel** each frame's masks with the global ids -> consistent
   ``semantic_instance/`` maps, plus a per-3D-point label vector for the
   fair 3D-metric comparison against off-the-shelf methods.

The association itself is pure (no IO) and unit-tested; the COLMAP/projection IO
lives in :mod:`worldsplat_prune.prep.scene_io`.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ..segment_sam2 import IMAGE_EXTS
from . import scene_io as sio


# --------------------------------------------------------------------------- #
# pure association logic                                                        #
# --------------------------------------------------------------------------- #
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:            # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def roots(self) -> List[int]:
        return [self.find(i) for i in range(len(self.parent))]


def build_point_index(obs_points: Sequence[np.ndarray]) -> Dict[int, List[int]]:
    """Map each 3D point id -> list of observation indices that contain it."""
    index: Dict[int, List[int]] = defaultdict(list)
    for oi, pts in enumerate(obs_points):
        for pt in np.asarray(pts).reshape(-1).tolist():
            index[pt].append(oi)
    return index


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def associate_observations(
    obs_points: Sequence[np.ndarray],
    obs_colors: Optional[Sequence[np.ndarray]] = None,
    *,
    min_shared_points: int = 2,
    color_sim: float = 0.0,
    point_index: Optional[Dict[int, List[int]]] = None,
) -> List[int]:
    """Union-find over masks that share 3D points. Returns root index per obs.

    ``color_sim > 0`` enables the colour gate: two masks are only merged when
    their mean colours are at least that cosine-similar.
    """
    n = len(obs_points)
    uf = UnionFind(n)
    if n == 0:
        return []
    index = point_index if point_index is not None else build_point_index(obs_points)

    pair_counts: Dict[Tuple[int, int], int] = {}
    for _pt, members in index.items():
        if len(members) < 2:
            continue
        m = sorted(set(members))
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                key = (m[i], m[j])
                pair_counts[key] = pair_counts.get(key, 0) + 1

    for (a, b), count in pair_counts.items():
        if count < min_shared_points:
            continue
        if color_sim > 0.0 and obs_colors is not None:
            if _cos_sim(obs_colors[a], obs_colors[b]) < color_sim:
                continue
        uf.union(a, b)
    return uf.roots()


def _mask_centroids(obs_points: Sequence[np.ndarray], coords: np.ndarray) -> np.ndarray:
    cents = np.full((len(obs_points), 3), np.nan, dtype=np.float64)
    for i, pts in enumerate(obs_points):
        if len(pts):
            cents[i] = coords[np.asarray(pts)].mean(axis=0)
    return cents


def _intersection_size(a: np.ndarray, b: np.ndarray) -> int:
    """|a ∩ b| for two sorted unique index arrays."""
    if a.size == 0 or b.size == 0:
        return 0
    if a.size > b.size:
        a, b = b, a
    pos = np.searchsorted(b, a)
    pos = np.clip(pos, 0, b.size - 1)
    return int(np.count_nonzero(b[pos] == a))


def associate_by_overlap(
    obs_points: Sequence[np.ndarray],
    coords: np.ndarray,
    obs_colors: Optional[Sequence[np.ndarray]] = None,
    *,
    metric: str = "jaccard",
    min_score: float = 0.25,
    color_sim: float = 0.0,
    grid_cell: Optional[float] = None,
    neighbor_steps: int = 2,
) -> List[int]:
    """Union-find over masks using **point-set overlap**, not shared-point count.

    Fixes the collapse of the naive ``shared`` rule: a single 3D point seen by
    many per-frame region masks used to transitively merge everything into one
    blob. Here two masks merge only if their *whole* 3D point sets overlap enough:
    ``jaccard = |A∩B|/|A∪B|`` or ``overlap = |A∩B|/min(|A|,|B|)`` >= ``min_score``
    (plus the optional colour gate). Candidate pairs are restricted to spatially
    nearby masks (3D centroid grid sized to the masks' own extent) so the
    association stays cheap.
    """
    n = len(obs_points)
    uf = UnionFind(n)
    if n == 0:
        return []

    sizes = np.array([len(p) for p in obs_points], dtype=np.int64)
    cents = _mask_centroids(obs_points, coords)
    good = np.isfinite(cents).all(axis=1) & (sizes > 0)
    idx = np.nonzero(good)[0]
    if idx.size == 0:
        return uf.roots()

    # per-mask spatial radius (how far its own points spread from its centroid)
    radii = np.zeros(n, dtype=np.float64)
    for i in idx.tolist():
        d = coords[np.asarray(obs_points[i])] - cents[i]
        radii[i] = float(np.linalg.norm(d, axis=1).max()) if d.size else 0.0
    pt_extent = float(np.linalg.norm(coords.max(0) - coords.min(0))) or 1.0
    cell = grid_cell if grid_cell else max(float(np.median(radii[idx])), pt_extent / 256.0, 1e-9)

    grid: Dict[tuple, List[int]] = defaultdict(list)
    for i in idx.tolist():
        grid[tuple(np.floor(cents[i] / cell).astype(int))].append(i)

    steps = int(max(0, neighbor_steps))
    offs = [(dx, dy, dz)
            for dx in range(-steps, steps + 1)
            for dy in range(-steps, steps + 1)
            for dz in range(-steps, steps + 1)]
    for i in idx.tolist():
        key = tuple(np.floor(cents[i] / cell).astype(int))
        a = obs_points[i]
        for off in offs:
            for j in grid.get((key[0] + off[0], key[1] + off[1], key[2] + off[2]), ()):
                if j <= i:
                    continue
                inter = _intersection_size(a, obs_points[j])
                if metric == "overlap":
                    score = inter / min(sizes[i], sizes[j])
                else:  # jaccard
                    score = inter / (sizes[i] + sizes[j] - inter)
                if score < min_score:
                    continue
                if color_sim > 0.0 and obs_colors is not None:
                    if _cos_sim(obs_colors[i], obs_colors[j]) < color_sim:
                        continue
                uf.union(i, j)
    return uf.roots()


def assign_global_ids(
    roots: Sequence[int],
    obs_cam: Sequence[int],
    *,
    min_views: int = 2,
) -> List[int]:
    """Map connected components to global ids; drop clusters below ``min_views``.

    Larger clusters get lower ids (stable and dense). Dropped observations get id 0
    (void).
    """
    groups: Dict[int, List[int]] = defaultdict(list)
    for oi, r in enumerate(roots):
        groups[r].append(oi)

    gid_of_obs = [0] * len(roots)
    next_id = 1
    for _root, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        support = len({obs_cam[m] for m in members})
        if support < max(1, min_views):
            continue
        for m in members:
            gid_of_obs[m] = next_id
        next_id += 1
    return gid_of_obs


# --------------------------------------------------------------------------- #
# IO-driven fusion                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class MaskObs:
    cam_index: int
    mask_id: int
    n_points: int
    color: Optional[np.ndarray] = None


def _read_rgb(images_dir: Path, name: str) -> Optional[np.ndarray]:
    p = images_dir / name
    if p.exists():
        return np.array(Image.open(p).convert("RGB"))
    for cand in images_dir.rglob(f"{Path(name).stem}.*"):
        if cand.suffix.lower() in IMAGE_EXTS:
            return np.array(Image.open(cand).convert("RGB"))
    return None


def _collect_observations(scene: sio.ColmapScene, dataset_dir: Path,
                          labels_dir: Path, images_subdir: str,
                          color_sim: float):
    """One pass over the frames -> per-(frame, mask) 3D point sets (+ colour)."""
    images_dir = dataset_dir / images_subdir
    obs: List[MaskObs] = []
    obs_points: List[np.ndarray] = []
    obs_colors: List[np.ndarray] = []
    obs_cam: List[int] = []
    cams_with_labels = []

    for ci, cam in enumerate(scene.cameras):
        lp = labels_dir / f"{cam.stem}.png"
        if not lp.exists():
            continue
        lab = sio.read_id_map(lp)
        if lab.shape != (cam.height, cam.width):
            from .. import semantic_utils as su
            lab = su.resize_ids(lab, cam.height, cam.width)
        lab_pt = sio.points_at_pixels(lab, cam, scene.xyz)

        img = _read_rgb(images_dir, cam.name) if color_sim > 0.0 else None
        ids = np.unique(lab_pt)
        ids = ids[ids > 0]
        cams_with_labels.append(ci)
        for mid in ids.tolist():
            pts = np.nonzero(lab_pt == mid)[0]
            if pts.size == 0:
                continue
            color = None
            if img is not None:
                region = lab == mid
                if region.any():
                    color = img[region].reshape(-1, 3).mean(axis=0)
            obs.append(MaskObs(cam_index=ci, mask_id=int(mid),
                               n_points=int(pts.size), color=color))
            obs_points.append(pts)
            obs_colors.append(color if color is not None else np.zeros(3))
            obs_cam.append(ci)
    return obs, obs_points, obs_colors, obs_cam, cams_with_labels


def fuse_instances(
    dataset_dir: str | Path,
    *,
    labels_dir: str | Path = "semantic_instance",
    out_dir: str | Path = "semantic_instance_fused",
    images_subdir: str = "images",
    sparse_subdir: str = "sparse/0",
    min_shared_points: int = 2,
    min_views: int = 2,
    color_sim: float = 0.0,
    radius_fill: int = 0,
    metric: str = "jaccard",
    min_score: float = 0.25,
    summary_path: Optional[str | Path] = None,
) -> dict:
    """Fuse per-frame masks into consistent 3D instances.

    Association defaults to overlap-based (``metric="jaccard"``, ``min_score``)
    which avoids the single-linkage collapse of the old ``metric="shared"`` rule.
    Writes ``<dataset>/<out_dir>/<stem>.png`` (per-view consistent ids) and
    ``<dataset>/<out_dir>/point_labels.npy`` (per-COLMAP-point ids), plus a
    summary JSON. Returns the summary dict.
    """
    dataset_dir = Path(dataset_dir)
    labels_dir = labels_dir if Path(labels_dir).is_absolute() else dataset_dir / labels_dir
    out_dir = out_dir if Path(out_dir).is_absolute() else dataset_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = sio.load_colmap_scene(dataset_dir / sparse_subdir)
    print(f"[fuse] {len(scene.cameras)} cameras, {scene.xyz.shape[0]} points; "
          f"labels <- {labels_dir}", flush=True)

    obs, obs_points, obs_colors, obs_cam, labeled_cams = _collect_observations(
        scene, dataset_dir, labels_dir, images_subdir, color_sim)
    print(f"[fuse] {len(obs)} mask observations on "
          f"{len(labeled_cams)}/{len(scene.cameras)} labeled frames", flush=True)

    point_index = build_point_index(obs_points)
    if metric == "shared":
        roots = associate_observations(
            obs_points, obs_colors, min_shared_points=min_shared_points,
            color_sim=color_sim, point_index=point_index)
    else:
        print(f"[fuse] association: metric={metric} min_score={min_score} "
              f"color_sim={color_sim}", flush=True)
        roots = associate_by_overlap(
            obs_points, scene.xyz, obs_colors, metric=metric,
            min_score=min_score, color_sim=color_sim)
    gid_of_obs = assign_global_ids(roots, obs_cam, min_views=min_views)
    n_instances = max(gid_of_obs, default=0)

    # --- per-view relabel -------------------------------------------------- #
    by_cam: Dict[int, List[int]] = defaultdict(list)
    for oi in range(len(obs)):
        if gid_of_obs[oi] > 0:
            by_cam[obs[oi].cam_index].append(oi)
    written = 0
    for ci in labeled_cams:
        cam = scene.cameras[ci]
        lab = sio.read_id_map(labels_dir / f"{cam.stem}.png")
        if lab.shape != (cam.height, cam.width):
            from .. import semantic_utils as su
            lab = su.resize_ids(lab, cam.height, cam.width)
        new = np.zeros_like(lab)
        for oi in by_cam.get(ci, []):
            new[lab == obs[oi].mask_id] = gid_of_obs[oi]
        if radius_fill > 0:
            new = sio.assign_within_radius(new, radius_fill)
        sio.write_id_map(out_dir / f"{cam.stem}.png", new)
        written += 1

    # --- per-point labels (for the 3D-metric comparison) ------------------- #
    point_labels = np.zeros(scene.xyz.shape[0], dtype=np.int32)
    for pt, members in point_index.items():
        gids = [gid_of_obs[oi] for oi in members if gid_of_obs[oi] > 0]
        if not gids:
            continue
        vals, counts = np.unique(gids, return_counts=True)
        if (counts == counts.max()).sum() > 1:      # ambiguous -> void
            continue
        point_labels[pt] = int(vals[counts.argmax()])
    np.save(out_dir / "point_labels.npy", point_labels)

    support = np.zeros(n_instances + 1, dtype=int)
    sizes = np.zeros(n_instances + 1, dtype=int)
    for oi, g in enumerate(gid_of_obs):
        if g > 0:
            sizes[g] += obs[oi].n_points
    for g in range(1, n_instances + 1):
        support[g] = len({obs_cam[oi] for oi in range(len(obs)) if gid_of_obs[oi] == g})

    inst_sizes = sizes[1:]
    total_pts = int(sizes[1:].sum()) or 1
    largest_frac = float(inst_sizes.max() / total_pts) if inst_sizes.size else 0.0
    summary = {
        "dataset": str(dataset_dir),
        "labels_dir": str(labels_dir),
        "out_dir": str(out_dir),
        "cameras": len(scene.cameras),
        "labeled_frames": len(labeled_cams),
        "masks": len(obs),
        "instances": int(n_instances),
        "views_written": written,
        "kept_obs": int(sum(1 for g in gid_of_obs if g > 0)),
        "metric": metric,
        "min_score": min_score,
        "min_shared_points": min_shared_points,
        "min_views": min_views,
        "color_sim": color_sim,
        "largest_instance_frac": round(largest_frac, 3),
        "instance_size_p50": int(np.percentile(inst_sizes, 50)) if inst_sizes.size else 0,
        "instance_size_p90": int(np.percentile(inst_sizes, 90)) if inst_sizes.size else 0,
        "support_hist": np.bincount(support[1:]).tolist() if n_instances else [],
        "points_labeled": int((point_labels > 0).sum()),
    }
    sp = Path(summary_path) if summary_path else out_dir / "fusion_summary.json"
    sp.write_text(json.dumps(summary, indent=2))
    print(f"[fuse] {n_instances} instances | {summary['kept_obs']}/{len(obs)} masks kept | "
          f"largest={largest_frac:.0%} of points | {summary['points_labeled']} points labeled "
          f"-> {out_dir}", flush=True)
    # collapse guard: a single (or dominant) instance means association failed and
    # pruning cannot discriminate on this labeling.
    if n_instances <= 1:
        print("[fuse] WARNING: only one instance -- association collapsed. "
              "Try metric=overlap, a lower --min_score, or --no_fill_void masks.",
              flush=True)
    elif largest_frac > 0.6:
        print(f"[fuse] WARNING: one instance covers {largest_frac:.0%} of labeled points; "
              "the partition is dominated by a single blob -- check --min_score/--color_sim.",
              flush=True)
    print(f"[fuse] summary -> {sp}", flush=True)
    return summary


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Minimal 3D instance fusion from per-frame masks")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--labels_dir", default="semantic_instance",
                    help="per-frame id maps to fuse (default <dataset>/semantic_instance)")
    ap.add_argument("--out_dir", default="semantic_instance_fused")
    ap.add_argument("--images_subdir", default="images")
    ap.add_argument("--sparse_subdir", default="sparse/0")
    ap.add_argument("--min_shared_points", type=int, default=2,
                    help="only for --metric shared: points two masks must share to merge")
    ap.add_argument("--metric", default="jaccard", choices=["jaccard", "overlap", "shared"],
                    help="mask association: point-set (jaccard/overlap) or shared-count")
    ap.add_argument("--min_score", type=float, default=0.25,
                    help="min point-set Jaccard/overlap to merge (jaccard/overlap metrics)")
    ap.add_argument("--min_views", type=int, default=2,
                    help="drop instances seen in fewer than this many frames")
    ap.add_argument("--color_sim", type=float, default=0.0,
                    help="cosine colour-similarity gate for merges (0 = off)")
    ap.add_argument("--radius_fill", type=int, default=0)
    args = ap.parse_args(argv)
    fuse_instances(
        args.dataset_dir, labels_dir=args.labels_dir, out_dir=args.out_dir,
        images_subdir=args.images_subdir, sparse_subdir=args.sparse_subdir,
        min_shared_points=args.min_shared_points, min_views=args.min_views,
        color_sim=args.color_sim, radius_fill=args.radius_fill,
        metric=args.metric, min_score=args.min_score)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
