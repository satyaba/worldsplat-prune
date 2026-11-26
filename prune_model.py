"""Post-reconstruction pruning driver: label → vote → prune → PLY (§3–5).

Loads an already-trained 3DGS model, lifts the per-view labels onto its Gaussians
(:mod:`worldsplat_prune.vote`), plans a semantic-aware pruning pass
(:mod:`worldsplat_prune.prune`) and writes a pruned PLY (plus the plan report) so
:mod:`eval_worldsplat` can score it.

    # semantic arm: keep 50%
    python prune_model.py -m <model_dir> --label_adapter composite \
        --keep_frac 0.5 --out_dir <model_dir>/pruned_semantic

    # geometry-only control (no band, no purity)
    python prune_model.py -m <model_dir> --label_adapter composite \
        --keep_frac 0.5 --no_band --no_purity --out_dir <model_dir>/pruned_geom

    # structure bar
    python prune_model.py -m <model_dir> --label_adapter composite \
        --keep_frac 0.5 --random --out_dir <model_dir>/pruned_random

    # plan only, no PLY
    python prune_model.py -m <model_dir> --label_adapter composite --dry_run
"""

from __future__ import annotations

import json
import os
import shutil
from argparse import ArgumentParser

import torch
from torch import nn

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state

from worldsplat_prune import prune as prune_mod
from worldsplat_prune.datasets import get_adapter
from worldsplat_prune.semantic_store import SemanticStore
from worldsplat_prune.vote import vote_classes_over_views


def _fill_model_defaults(dataset) -> None:
    """Give a vanilla (upstream graphdeco) cfg_args the fields this fork reads."""
    for k, v in {"images": "images", "depths": "", "resolution": -1,
                 "white_background": False, "train_test_exp": False,
                 "eval": False, "data_device": "cuda"}.items():
        if not hasattr(dataset, k):
            setattr(dataset, k, v)


def _subset_model(src: GaussianModel, keep: torch.Tensor) -> GaussianModel:
    """A new ``GaussianModel`` holding only the kept Gaussians (for save_ply)."""
    out = GaussianModel(src.max_sh_degree)
    out.active_sh_degree = src.active_sh_degree
    out._xyz = nn.Parameter(src._xyz[keep].detach().clone())
    out._features_dc = nn.Parameter(src._features_dc[keep].detach().clone())
    out._features_rest = nn.Parameter(src._features_rest[keep].detach().clone())
    out._opacity = nn.Parameter(src._opacity[keep].detach().clone())
    out._scaling = nn.Parameter(src._scaling[keep].detach().clone())
    out._rotation = nn.Parameter(src._rotation[keep].detach().clone())
    return out


def _default_tag(a) -> str:
    kind = "random" if a.random else ("geom" if (a.no_band and a.no_purity) else "semantic")
    return f"pruned_{kind}_k{a.keep_frac:g}"


def run(dataset, iteration, pipe, a, out_dir=None) -> dict:
    dataset.depths = ""
    _fill_model_defaults(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    cams = scene.getTrainCameras()
    print(f"[prune] {gaussians.get_xyz.shape[0]} gaussians | "
          f"{len(cams)} train cameras | extent {scene.cameras_extent:.3f}")

    adapter = get_adapter(a.label_adapter, dataset.source_path,
                          {"label_dir": a.label_dir or None, "min_area": a.min_seg_area})
    store = SemanticStore(adapter, device="cuda")
    print(f"[prune] label adapter: {adapter.meta()}")

    bundle = vote_classes_over_views(
        gaussians, cams, store,
        visibility=a.visibility, vote=a.vote, dt_norm=a.dt_norm,
        min_views=a.min_views, occlusion=not a.no_occlusion,
        depth_tol=a.depth_tol, mask_margin=a.mask_margin,
        camera_order="sorted", pipe=pipe)
    print("[prune] " + bundle.summary())

    plan = prune_mod.plan_pruning(
        gaussians.get_opacity, gaussians.get_scaling,
        boundary_dist=bundle.boundary_dist, seg=bundle.seg, purity=bundle.purity,
        extent=scene.cameras_extent,
        keep_frac=a.keep_frac, n_target=a.n_target,
        band=a.band, epsilon=a.epsilon, strength=a.strength, curve=a.curve,
        use_band=(not a.no_band) and (not a.random), use_purity=(not a.no_purity),
        allocation=a.allocation, target=a.target_, random_control=a.random,
        seed=a.seed)

    report = plan.report()
    report["vote"] = {k: bundle.meta[k] for k in
                      ("visibility", "vote", "occlusion", "min_views",
                       "n_visible", "n_labeled", "mean_purity")}
    print(f"[prune] plan: {report}")

    if not a.dry_run:
        out_dir = str(out_dir or a.out_dir or
                      os.path.join(dataset.model_path, _default_tag(a)))
        os.makedirs(os.path.join(out_dir, "point_cloud",
                                 f"iteration_{scene.loaded_iter}"), exist_ok=True)
        pruned = _subset_model(gaussians, plan.keep)
        pruned.save_ply(os.path.join(out_dir, "point_cloud",
                                     f"iteration_{scene.loaded_iter}", "point_cloud.ply"))
        # a valid model dir needs the trained cfg for eval_worldsplat to read
        cfg_src = os.path.join(dataset.model_path, "cfg_args")
        if os.path.exists(cfg_src):
            shutil.copyfile(cfg_src, os.path.join(out_dir, "cfg_args"))
        with open(os.path.join(out_dir, "prune_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"[prune] wrote {out_dir} "
              f"({plan.n_after} gaussians, iteration {scene.loaded_iter})")
        report["out_dir"] = out_dir

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[prune] wrote {a.json_out}")
    return report


if __name__ == "__main__":
    parser = ArgumentParser(description="WorldSplat-Prune post-reconstruction pruning")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    # label source
    parser.add_argument("--label_adapter", default=None, type=str,
                        help="worldsplat_prune/datasets adapter (e.g. composite)")
    parser.add_argument("--label_dir", default=None, type=str)
    parser.add_argument("--min_seg_area", default=None, type=int)
    # vote pass (§3)
    parser.add_argument("--visibility", default=None, type=str, choices=["render", "project"])
    parser.add_argument("--vote", default=None, type=str, choices=["first", "majority"])
    parser.add_argument("--dt_norm", default=None, type=float)
    parser.add_argument("--min_views", default=None, type=int)
    parser.add_argument("--no_occlusion", action="store_true")
    parser.add_argument("--depth_tol", default=None, type=float)
    parser.add_argument("--mask_margin", default=None, type=float)
    # pruning (§4-5)
    parser.add_argument("--keep_frac", default=0.5, type=float)
    parser.add_argument("--n_target", default=None, type=int)
    parser.add_argument("--band", default=None, type=float)
    parser.add_argument("--epsilon", default=None, type=float)
    parser.add_argument("--strength", default=None, type=float)
    parser.add_argument("--curve", default=None, type=float)
    parser.add_argument("--no_band", action="store_true")
    parser.add_argument("--no_purity", action="store_true")
    parser.add_argument("--allocation", default=None, type=str,
                        choices=["global", "per_segment"])
    parser.add_argument("--target", default=None, type=int,
                        help="consistent instance id to keep whole (mode b)")
    parser.add_argument("--random", action="store_true",
                        help="random structure bar (random allocation; band protect off)")
    parser.add_argument("--seed", default=0, type=int)
    # output
    parser.add_argument("--out_dir", default=None, type=str)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--json_out", default=None, type=str)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    # fill vote/prune defaults where the CLI left them None
    def _d(name, val):
        cur = getattr(args, name, None)
        if cur is None:
            setattr(args, name, val)
        return getattr(args, name)

    _d("visibility", "render"); _d("vote", "first"); _d("dt_norm", 100.0)
    _d("min_views", 3); _d("depth_tol", 0.02); _d("mask_margin", 0.0)
    _d("band", prune_mod.DEFAULT_BAND); _d("epsilon", prune_mod.DEFAULT_EPSILON)
    _d("strength", prune_mod.DEFAULT_STRENGTH); _d("curve", prune_mod.DEFAULT_CURVE)
    _d("allocation", "global")
    args.label_adapter = args.label_adapter or "replica"
    args.min_seg_area = args.min_seg_area or 0
    args.target_ = args.target

    safe_state(args.quiet)
    print("Pruning " + args.model_path)
    run(model.extract(args), args.iteration, pipeline.extract(args), args,
        out_dir=args.out_dir)
