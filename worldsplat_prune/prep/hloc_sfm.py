"""HLOC SfM -> a 3DGS-ready COLMAP scene.

The reference 3DGS pipeline builds its SfM with COLMAP's own ``convert.py``.
This module reaches the *same* on-disk contract with **HLOC** as the front-end
(learned features/matchers, optional sequential pairing) and COLMAP/pycolmap only
for the final undistortion step::

    <out>/images/        # undistorted frames
    <out>/sparse/0/      # cameras.bin / images.bin / points3D.bin, PINHOLE camera

The PINHOLE requirement is not cosmetic: ``scene/dataset_readers.py`` asserts
PINHOLE / SIMPLE_PINHOLE, and the rasterizer assumes an undistorted pinhole. HLOC
defaults to ``SIMPLE_RADIAL``, so by default we feature-extract with radial
distortion and then **undistort** (exactly what ``convert.py`` does). If your
capture is already rectified, pass ``--no_undistort`` (and/or
``--camera_model PINHOLE``) to skip it.

Colab (from the repo root):

    !pip install -q hloc  # or: git+https://github.com/cvg/Hierarchical-Localization

    python -m worldsplat_prune.prep.hloc_sfm \\
        --images "$HOME_DIR/raw/mipnerf360/garden/images" \\
        --out    "$DATASET_ROOT/mipnerf360/garden" \\
        --feature superpoint_aachen --matcher superglue

Heavy deps (hloc, pycolmap) are imported lazily so this module can be imported
(e.g. for its helpers) without them installed.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

from ..segment_sam2 import IMAGE_EXTS


def _image_list(images_dir: Path) -> list:
    return sorted(p.relative_to(images_dir).as_posix()
                  for p in images_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def run_hloc_sfm(
    images_dir: str | Path,
    work_dir: str | Path,
    *,
    feature: str = "superpoint_aachen",
    matcher: str = "superglue",
    camera_model: str = "SIMPLE_RADIAL",
    pairs: str = "exhaustive",
    window_size: int = 20,
    max_image_size: Optional[int] = None,
):
    """Extract features, pair, match and reconstruct.

    Returns ``(reconstruction, features, matches)``: the pycolmap
    :class:`Reconstruction` and the HLOC feature/match HDF5 paths. Everything is
    cached under ``work_dir``.
    """
    from hloc import (extract_features, match_features, pairs_from_exhaustive,
                      pairs_from_sequential, reconstruction)

    images_dir = Path(images_dir)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    image_list = _image_list(images_dir)
    if not image_list:
        raise FileNotFoundError(f"no images under {images_dir}")

    feature_conf = dict(extract_features.confs[feature])
    feature_conf["camera_model"] = camera_model
    if max_image_size:
        feature_conf["preprocessing"] = {"resize_max": max_image_size}
    matcher_conf = dict(match_features.confs[matcher])

    print(f"[hloc] features={feature} (camera_model={camera_model}) "
          f"matcher={matcher} images={len(image_list)}", flush=True)
    features = extract_features.main(feature_conf, images_dir, work_dir,
                                     image_list=image_list)

    sfm_pairs = work_dir / "pairs.txt"
    if pairs == "sequential":
        pairs_from_sequential.main(sfm_pairs, features=features,
                                   image_list=image_list, window_size=window_size)
    else:
        pairs_from_exhaustive.main(sfm_pairs, image_list=image_list)
    matches = match_features.main(matcher_conf, sfm_pairs, features=features,
                                  export_dir=work_dir)

    sfm_dir = work_dir / "sparse"
    rec = reconstruction.main(sfm_dir, images_dir, sfm_pairs, features, matches,
                              image_list=image_list)
    if rec is None or len(rec.registered_images) == 0:
        raise RuntimeError(
            "HLOC reconstruction registered no images; try more/better "
            "features, `--pairs sequential`, or check the input frames.")
    print(f"[hloc] registered {len(rec.registered_images)}/{len(image_list)} "
          f"images, {len(rec.points3D)} points", flush=True)
    return rec, features, matches


def _place_images(src: Path, dst: Path, link: bool) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            target = dst / p.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            if link:
                try:
                    if target.exists():
                        target.unlink()
                    target.symlink_to(p.resolve())
                    continue
                except OSError:
                    pass  # fall back to a copy (e.g. Windows without dev mode)
            shutil.copy2(p, target)


def _undistort(rec, images_dir: Path, out_dir: Path) -> None:
    """Undistort into ``out_dir`` (images/ + sparse/0, PINHOLE)."""
    import pycolmap

    model_dir = out_dir / "_sfm_model"
    sparse0 = model_dir / "sparse" / "0"
    sparse0.mkdir(parents=True, exist_ok=True)
    rec.write(str(sparse0))

    # pycolmap.undistort_images(output_dir, input_sfm_dir, image_dir): the
    # canonical directory form (matches COLMAP's `image_undistorter`).
    try:
        pycolmap.undistort_images(str(out_dir), str(model_dir), str(images_dir))
    except (TypeError, AttributeError):
        # Older/newer binding: fall back to the COLMAP CLI.
        cmd = ["colmap", "image_undistorter",
               "--image_path", str(images_dir),
               "--input_path", str(sparse0),
               "--output_path", str(out_dir),
               "--output_type", "COLMAP"]
        print("[hloc] pycolmap.undistort_images unavailable; "
              f"running: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(model_dir, ignore_errors=True)


def build_colmap_scene(
    images_dir: str | Path,
    out_dir: str | Path,
    *,
    feature: str = "superpoint_aachen",
    matcher: str = "superglue",
    camera_model: str = "SIMPLE_RADIAL",
    pairs: str = "exhaustive",
    window_size: int = 20,
    max_image_size: Optional[int] = None,
    undistort: bool = True,
    link_images: bool = True,
    work_dir: Optional[str | Path] = None,
) -> Path:
    """Build ``<out_dir>/{images, sparse/0}`` from raw ``images_dir``."""
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    work = Path(work_dir) if work_dir else out_dir / "_hloc"
    work.mkdir(parents=True, exist_ok=True)

    rec, _, _ = run_hloc_sfm(
        images_dir, work, feature=feature, matcher=matcher,
        camera_model=camera_model, pairs=pairs, window_size=window_size,
        max_image_size=max_image_size)

    do_undistort = undistort and camera_model.upper() != "PINHOLE"
    if do_undistort:
        print(f"[hloc] undistorting -> {out_dir} (image_undistorter)", flush=True)
        _undistort(rec, images_dir, out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        sparse0 = out_dir / "sparse" / "0"
        sparse0.mkdir(parents=True, exist_ok=True)
        rec.write(str(sparse0))
        _place_images(images_dir, out_dir / "images", link_images)

    # Sanity: the model must be PINHOLE/SIMPLE_PINHOLE for 3DGS.
    import pycolmap
    rec_out = pycolmap.Reconstruction(str(out_dir / "sparse" / "0"))
    models = {c.model.name for c in rec_out.cameras.values()}
    ok = models <= {"PINHOLE", "SIMPLE_PINHOLE"}
    print(f"[hloc] done: {out_dir} | cameras={sorted(models)} "
          f"| {'OK' if ok else 'NOT 3DGS-COMPATIBLE'}", flush=True)
    if not ok:
        print("[hloc] WARNING: 3DGS needs PINHOLE/SIMPLE_PINHOLE; "
              "re-run with distortion+undistort (default) or --camera_model PINHOLE.",
              file=sys.stderr)
    return out_dir


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="HLOC SfM -> 3DGS COLMAP scene")
    ap.add_argument("--images", required=True, help="raw image directory")
    ap.add_argument("--out", required=True, help="output dataset dir (images/ + sparse/0/)")
    ap.add_argument("--work_dir", default=None, help="scratch dir (default <out>/_hloc)")
    ap.add_argument("--feature", default="superpoint_aachen")
    ap.add_argument("--matcher", default="superglue")
    ap.add_argument("--camera_model", default="SIMPLE_RADIAL")
    ap.add_argument("--pairs", default="exhaustive", choices=["exhaustive", "sequential"])
    ap.add_argument("--window_size", type=int, default=20)
    ap.add_argument("--max_image_size", type=int, default=None)
    ap.add_argument("--no_undistort", action="store_true")
    ap.add_argument("--copy_images", action="store_true",
                    help="copy instead of symlink (only used with --no_undistort)")
    args = ap.parse_args(argv)

    build_colmap_scene(
        args.images, args.out, feature=args.feature, matcher=args.matcher,
        camera_model=args.camera_model, pairs=args.pairs,
        window_size=args.window_size, max_image_size=args.max_image_size,
        undistort=not args.no_undistort, link_images=not args.copy_images,
        work_dir=args.work_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
