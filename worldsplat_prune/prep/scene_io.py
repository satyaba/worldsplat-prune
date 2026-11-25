"""COLMAP scene IO + projection for the instance-fusion tools.

``worldsplat_prune.prep`` deliberately does **not** import torch: the ``scene`` package pulls
torch in via its ``__init__``, so this module re-implements the small binary/text
readers (mirroring ``scene/colmap_loader.py``) and stays numpy/cv2/PIL-only. That
keeps dataset preparation testable on a CPU-only machine.

Conventions match the trainer exactly:

* world-to-camera: ``x_cam = R_wc @ X + t_wc`` with ``R_wc = qvec2rotmat(qvec)``
  and ``t_wc = tvec`` (the repo stores ``R = qvec2rotmat^T`` and
  ``getWorld2View2`` transposes it back, so this is the effective transform);
* pixels: ``u = fx * x/z + cx``, ``v = fy * y/z + cy`` (PINHOLE params
  ``[fx, fy, cx, cy]``; SIMPLE_PINHOLE ``[f, cx, cy]``).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from .. import semantic_utils as su

# --------------------------------------------------------------------------- #
# data model                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class Camera:
    camera_id: int
    image_id: int
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R_wc: np.ndarray          # (3, 3) world -> camera rotation
    t_wc: np.ndarray          # (3,)   world -> camera translation
    model: str = "PINHOLE"    # COLMAP camera model name

    @property
    def stem(self) -> str:
        return Path(self.name).stem


@dataclass
class ColmapScene:
    cameras: List[Camera]
    xyz: np.ndarray           # (P, 3) float64
    rgb: np.ndarray           # (P, 3) uint8
    by_stem: Dict[str, Camera] = field(default_factory=dict)

    def __post_init__(self):
        self.by_stem = {c.stem: c for c in self.cameras}


# --------------------------------------------------------------------------- #
# COLMAP readers (self-contained; mirrors scene/colmap_loader.py)              #
# --------------------------------------------------------------------------- #
def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    q = qvec
    return np.array([
        [1 - 2 * q[2] ** 2 - 2 * q[3] ** 2,
         2 * q[1] * q[2] - 2 * q[0] * q[3],
         2 * q[3] * q[1] + 2 * q[0] * q[2]],
        [2 * q[1] * q[2] + 2 * q[0] * q[3],
         1 - 2 * q[1] ** 2 - 2 * q[3] ** 2,
         2 * q[2] * q[3] - 2 * q[0] * q[1]],
        [2 * q[3] * q[1] - 2 * q[0] * q[2],
         2 * q[2] * q[3] + 2 * q[0] * q[1],
         1 - 2 * q[1] ** 2 - 2 * q[2] ** 2]])


def _next(fid, n, fmt):
    return struct.unpack("<" + fmt, fid.read(n))


def _read_cameras_binary(path: Path) -> dict:
    cams = {}
    with open(path, "rb") as fid:
        num = _next(fid, 8, "Q")[0]
        for _ in range(num):
            cid, model_id, w, h = _next(fid, 24, "iiQQ")
            num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12,
                          7: 5, 8: 4, 9: 5, 10: 12}[model_id]
            params = _next(fid, 8 * num_params, "d" * num_params)
            model_name = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL",
                          3: "RADIAL", 4: "OPENCV", 5: "OPENCV_FISHEYE",
                          6: "FULL_OPENCV", 7: "FOV", 8: "SIMPLE_RADIAL_FISHEYE",
                          9: "RADIAL_FISHEYE", 10: "THIN_PRISM_FISHEYE"}[model_id]
            cams[cid] = (model_name, int(w), int(h), np.array(params, dtype=np.float64))
    return cams


def _read_images_binary(path: Path) -> dict:
    imgs = {}
    with open(path, "rb") as fid:
        num = _next(fid, 8, "Q")[0]
        for _ in range(num):
            props = _next(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5], dtype=np.float64)
            tvec = np.array(props[5:8], dtype=np.float64)
            camera_id = props[8]
            name = ""
            ch = _next(fid, 1, "c")[0]
            while ch != b"\x00":
                name += ch.decode("utf-8")
                ch = _next(fid, 1, "c")[0]
            n2d = _next(fid, 8, "Q")[0]
            _next(fid, 24 * n2d, "ddq" * n2d)
            imgs[image_id] = (qvec, tvec, camera_id, name)
    return imgs


def _read_points3D_binary(path: Path) -> tuple:
    with open(path, "rb") as fid:
        num = _next(fid, 8, "Q")[0]
        xyz = np.empty((num, 3), dtype=np.float64)
        rgb = np.empty((num, 3), dtype=np.uint8)
        for i in range(num):
            props = _next(fid, 43, "QdddBBBd")
            xyz[i] = props[1:4]
            rgb[i] = props[4:7]
            track_len = _next(fid, 8, "Q")[0]
            _next(fid, 8 * track_len, "ii" * track_len)
    return xyz, rgb


def _read_cameras_text(path: Path) -> dict:
    cams = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        e = line.split()
        cams[int(e[0])] = (e[1], int(e[2]), int(e[3]), np.array(list(map(float, e[4:]))))
    return cams


def _read_images_text(path: Path) -> dict:
    imgs = {}
    lines = [l.rstrip("\n") for l in path.read_text().splitlines()
             if not l.startswith("#")]
    for i in range(0, len(lines), 2):  # 2 lines per image: pose, then points2D
        e = lines[i].split()
        if not e:
            continue
        imgs[int(e[0])] = (np.array(list(map(float, e[1:5]))),
                           np.array(list(map(float, e[5:8]))), int(e[8]), e[9])
    return imgs


def _read_points3D_text(path: Path) -> tuple:
    xs, cs = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        e = line.split()
        xs.append(list(map(float, e[1:4])))
        cs.append(list(map(int, e[4:7])))
    return np.array(xs, dtype=np.float64), np.array(cs, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# loading                                                                      #
# --------------------------------------------------------------------------- #
def _intrinsics(model: str, params: np.ndarray, w: int, h: int):
    m = model.upper()
    if m == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif m == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        fx = fy = f
    else:
        raise ValueError(
            f"camera model {model!r} not supported; the rasterizer needs an "
            "undistorted PINHOLE / SIMPLE_PINHOLE scene (run worldsplat_prune.prep.hloc_sfm "
            "or convert.py to undistort).")
    return float(fx), float(fy), float(cx), float(cy)


def load_colmap_scene(sparse_dir: str | Path) -> ColmapScene:
    """Load ``sparse/0`` (binary preferred, text fallback) into a ColmapScene."""
    sparse_dir = Path(sparse_dir)
    if (sparse_dir / "cameras.bin").exists():
        cams_raw = _read_cameras_binary(sparse_dir / "cameras.bin")
        imgs_raw = _read_images_binary(sparse_dir / "images.bin")
        xyz, rgb = _read_points3D_binary(sparse_dir / "points3D.bin")
    elif (sparse_dir / "cameras.txt").exists():
        cams_raw = _read_cameras_text(sparse_dir / "cameras.txt")
        imgs_raw = _read_images_text(sparse_dir / "images.txt")
        xyz, rgb = _read_points3D_text(sparse_dir / "points3D.txt")
    else:
        raise FileNotFoundError(f"no COLMAP model found under {sparse_dir}")

    cameras: List[Camera] = []
    for image_id, (qvec, tvec, camera_id, name) in imgs_raw.items():
        model, w, h, params = cams_raw[camera_id]
        fx, fy, cx, cy = _intrinsics(model, params, w, h)
        cameras.append(Camera(
            camera_id=int(camera_id), image_id=int(image_id), name=name,
            width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy,
            R_wc=qvec2rotmat(qvec), t_wc=np.asarray(tvec, dtype=np.float64),
            model=model))
    cameras.sort(key=lambda c: c.name)
    return ColmapScene(cameras=cameras, xyz=xyz, rgb=rgb)


# --------------------------------------------------------------------------- #
# projection                                                                   #
# --------------------------------------------------------------------------- #
def project_points(xyz: np.ndarray, cam: Camera) -> tuple:
    """Project world points into ``cam``. Returns ``(u, v, z, valid)``.

    ``valid`` is True where the point is in front of the camera (``z > 0``);
    pixel bounds are *not* applied here (callers clamp/index).
    """
    xc = xyz @ cam.R_wc.T + cam.t_wc
    z = xc[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = cam.fx * xc[:, 0] / safe + cam.cx
    v = cam.fy * xc[:, 1] / safe + cam.cy
    return u, v, z, z > 1e-6


def points_at_pixels(label_map: np.ndarray, cam: Camera, xyz: np.ndarray) -> np.ndarray:
    """Per-3D-point label id sampled at its projection (0 where void/offscreen)."""
    H, W = label_map.shape
    u, v, z, valid = project_points(xyz, cam)
    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)
    inb = valid & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    out = np.zeros(xyz.shape[0], dtype=np.int64)
    if inb.any():
        out[inb] = label_map[vi[inb], ui[inb]]
    return out


def point_labels_from_views(scene: "ColmapScene", labels_dir: str | Path,
                            mode: str = "first", min_votes: int = 1,
                            transform=None) -> np.ndarray:
    """Lift per-view id maps onto the COLMAP points.

    ``mode="first"``   : first labelled view wins (the raw/unfused behaviour).
    ``mode="majority"``: majority vote across views (0 on ties or below
                         ``min_votes``); this is how a 3D GT/instance labelling is
                         built from per-view maps.

    Per-view maps may be PNG id maps or int32 ``.npy`` maps (e.g. composite
    ``(class, instance)`` maps, which exceed the 16-bit PNG range). An optional
    ``transform`` is applied to each loaded map before lifting — e.g.
    ``transform=decode_class`` to lift the class half of a composite map.
    """
    labels_dir = Path(labels_dir)
    P = scene.xyz.shape[0]

    def _lift(cam):
        lab = read_view_map(labels_dir, cam.stem)
        if lab is None:
            return None
        if transform is not None:
            lab = transform(lab)
        return points_at_pixels(lab, cam, scene.xyz)

    if mode == "first":
        out = np.zeros(P, dtype=np.int32)
        assigned = np.zeros(P, dtype=bool)
        for cam in scene.cameras:
            lab = _lift(cam)
            if lab is None:
                continue
            new = (lab > 0) & (~assigned)
            out[new] = lab[new]
            assigned |= new
        return out

    # majority: gather (point, label) votes across all views in one pass
    LARGE = 1 << 16
    pts_list, lab_list = [], []
    for cam in scene.cameras:
        lab = _lift(cam)
        if lab is None:
            continue
        idx = np.nonzero(lab > 0)[0]
        if idx.size:
            pts_list.append(idx.astype(np.int64))
            lab_list.append(lab[idx].astype(np.int64))
    if not pts_list:
        return np.zeros(P, dtype=np.int32)
    pts = np.concatenate(pts_list)
    labs = np.concatenate(lab_list)
    uniq, counts = np.unique(pts * LARGE + labs, return_counts=True)
    u_pt, u_lab = np.divmod(uniq, LARGE)
    # sort by point asc, count desc -> the first row per point is its winner
    order = np.lexsort((-counts, u_pt))
    u_pt, u_lab, counts = u_pt[order], u_lab[order], counts[order]
    first = np.ones(u_pt.size, dtype=bool)
    first[1:] = u_pt[1:] != u_pt[:-1]

    out = np.zeros(P, dtype=np.int32)
    pt_max = np.zeros(P, dtype=np.int64)
    pt_max[u_pt[first]] = counts[first]
    # a point is decided only when its top count is unique (no tie) and >= min_votes
    is_max = counts == pt_max[u_pt]
    n_max = np.zeros(P, dtype=np.int64)
    np.add.at(n_max, u_pt, is_max.astype(np.int64))
    keep = first & (n_max[u_pt] == 1) & (counts >= min_votes)
    out[u_pt[keep]] = u_lab[keep]
    return out


# --------------------------------------------------------------------------- #
# id-map IO                                                                    #
# --------------------------------------------------------------------------- #
def read_id_map(path: str | Path) -> np.ndarray:
    """Read a label image (palette/8-bit/16-bit safe) as ``int32 [H, W]``."""
    return su.read_ids(path)


def write_id_map(path: str | Path, ids: np.ndarray) -> None:
    """Write an id map as a 16-bit PNG (supports up to 65535 instances/frame)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(ids).astype(np.uint16)).save(path)


def read_view_map(labels_dir: str | Path, stem: str) -> np.ndarray | None:
    """Read one per-view label map: ``<stem>.npy`` (int32) or ``<stem>.png``.

    Returns ``None`` when neither exists. The ``.npy`` form carries composite
    ``(class, instance)`` maps whose ids exceed the 16-bit PNG range.
    """
    labels_dir = Path(labels_dir)
    npy = labels_dir / f"{stem}.npy"
    if npy.exists():
        return np.load(npy).astype(np.int32)
    png = labels_dir / f"{stem}.png"
    if png.exists():
        return read_id_map(png)
    return None


def assign_within_radius(label_map: np.ndarray, radius: int) -> np.ndarray:
    """Extend labels into nearby void pixels (depth-free "splat" of 2D labels).

    Unlike :func:`worldsplat_prune.semantic_utils.fill_void_nearest`, this only fills pixels
    within ``radius`` of a labelled pixel, so empty background is left as void.
    """
    if radius <= 0:
        return label_map.astype(np.int32)
    label_map = label_map.astype(np.int32)
    void = label_map == 0
    if not void.any() or void.all():
        return label_map
    # void=1, labelled=0 -> distance/label of the nearest labelled pixel.
    dist, nearest = cv2.distanceTransformWithLabels(
        void.astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(int(nearest.max()) + 1, dtype=np.int32)
    lut[nearest[~void]] = label_map[~void]
    out = label_map.copy()
    fill = void & (dist <= radius)
    out[fill] = lut[nearest[fill]]
    return out


def splat_point_labels_to_views(scene: ColmapScene, point_labels: np.ndarray,
                                out_dir: str | Path, radius: int = 2) -> int:
    """Rasterise **3D point labels** into per-view id maps.

    This is the normalizer that lets *any* method producing per-point-cloud
    instance labels (MaskClustering, SAMPro3D, our fusion, ...) be fed to the
    per-view ``semantic_instance/`` interface WorldSplat-Prune consumes. Points on screen are
    painted far-to-near so the nearest surface wins; labels are then dilated into
    a ``radius``-px neighbourhood. Returns the number of views written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    point_labels = np.asarray(point_labels)
    written = 0
    for cam in scene.cameras:
        u, v, z, valid = project_points(scene.xyz, cam)
        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        inb = valid & (ui >= 0) & (ui < cam.width) & (vi >= 0) & (vi < cam.height)
        sel = np.nonzero(inb & (point_labels > 0))[0]
        m = np.zeros((cam.height, cam.width), dtype=np.int32)
        if sel.size:
            order = sel[np.argsort(-z[sel])]      # far first, near last (occlusion)
            m[vi[order], ui[order]] = point_labels[order]
        m = assign_within_radius(m, radius)
        write_id_map(out_dir / f"{cam.stem}.png", m)
        written += 1
    return written
