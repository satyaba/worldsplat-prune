"""Smoke tests for the ``experiments`` harness (no GPU / torch required).

Covers the parts that are easy to break: the run manifest, the train-command
builder, the ``--extra`` token splitting, and the metrics row shaping.

    pytest tests/test_experiments.py
    python tests/test_experiments.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import runner, runs
from experiments.metrics import CSV_COLUMNS, to_row


def test_manifest_is_well_formed():
    assert runs.RUNS, "manifest is empty"
    for run_id, spec in runs.RUNS.items():
        assert run_id == spec.run_id
        assert spec.iterations > 0
    # every referenced method exists
    from experiments.methods import METHODS
    for spec in runs.RUNS.values():
        assert spec.method in METHODS, f"{spec.run_id}: unknown method {spec.method}"


def test_sanity_partition():
    full = {r.run_id for r in runs.full_runs()}
    sanity = {r.run_id for r in runs.sanity_runs()}
    assert not (full & sanity)
    assert full | sanity == set(runs.RUNS)


def test_build_train_cmd_order():
    from experiments.methods import get_method

    m = get_method("worldsplat_prune")
    cmd = runner.build_train_cmd(
        m, source="/data", model_dir="/out", iterations=3000,
        extra=("--position_lr_init", "0.00016"),
    )
    assert cmd[0].endswith("python") or "python" in cmd[0]
    assert cmd[1] == "train.py"
    # -s then -m
    assert cmd[cmd.index("-s") + 1] == "/data"
    assert cmd[cmd.index("-m") + 1] == "/out"
    # method common args + iterations are present
    for a in m.common_args:
        assert a in cmd
    assert "--iterations" in cmd and "3000" in cmd
    # extra comes last, after --iterations (argparse last-wins)
    assert cmd[-2:] == ["--position_lr_init", "0.00016"]


def test_split_extra_string_and_tokens():
    assert runner._split_extra(["--position_lr_init 0.00016 --lambda_dssim 0.3"]) == [
        "--position_lr_init", "0.00016", "--lambda_dssim", "0.3"]
    assert runner._split_extra(["--position_lr_init", "0.00016"]) == ["--position_lr_init", "0.00016"]
    assert runner._split_extra(None) == []
    assert runner._split_extra([]) == []


def test_run_overrides_append_after_spec_extra():
    captured = {}

    def fake_run_method(method_id, **kw):
        captured["method"] = method_id
        captured["extra"] = list(kw["extra"])
        return kw["model_dir"]

    original = runs.run_method
    runs.run_method = fake_run_method
    try:
        spec = runs.RunSpec(run_id="probe", method="worldsplat_prune",
                            extra=("--position_lr_init", "0.00016"))
        runs.run(spec, home_dir="/h", dataset_root="/ds",
                 extra="--position_lr_final 1e-6")
    finally:
        runs.run_method = original
    extra = captured["extra"]
    # spec's own tokens come first, overrides last
    assert extra[:2] == ["--position_lr_init", "0.00016"]
    assert extra[-2:] == ["--position_lr_final", "1e-6"]


def test_to_row_shapes_and_rounds():
    data = {
        "run_dir": "/out/base_30k",
        "iteration": 30000,
        "gaussians": 123456,
        "splits": {
            "train": {"psnr": 30.1234, "ssim": 0.912345, "lpips": 0.20123},
            "test": {"psnr": 28.4444, "ssim": 0.854321, "lpips": 0.31234,
                     "psnr_boundary": 25.123, "psnr_interior": 29.876},
        },
    }
    row = to_row(data, run_id="base_30k", name="baseline")
    assert row["Name"] == "baseline"
    assert row["Run ID"] == "base_30k"
    assert row["Iteration"] == 30000
    assert row["Test PSNR"] == 28.44
    assert row["Test SSIM"] == 0.8543
    assert row["Test PSNR band"] == 25.12
    assert row["Test PSNR interior"] == 29.88
    assert row["Gauss. count"] == 123456
    assert set(row) == set(CSV_COLUMNS)


def test_to_row_missing_band_metrics_stay_none():
    data = {"run_dir": "/out/base", "iteration": 30000, "gaussians": 10,
            "splits": {"test": {"psnr": 27.0, "ssim": 0.8, "lpips": 0.4}}}
    row = to_row(data)
    assert row["Test PSNR band"] is None
    assert row["Test PSNR interior"] is None


# --------------------------------------------------------------------------- #
# pruning-ablation manifest (§4-5)                                              #
# --------------------------------------------------------------------------- #
def _arm_spec(arm):
    for s in runs.PRUNE_RUNS.values():
        if s.arm == arm:
            return s
    raise AssertionError(f"no spec for arm {arm!r}")


def test_prune_manifest_well_formed():
    assert runs.PRUNE_RUNS, "prune manifest is empty"
    ids = [s.run_id for s in runs.PRUNE_RUNS.values()]
    assert len(ids) == len(set(ids))
    for rid, s in runs.PRUNE_RUNS.items():
        assert rid == s.run_id
        assert s.arm in runs.PRUNE_ARMS
        assert 0.0 < s.keep_frac <= 1.0
        assert 0.0 <= s.strength <= 1.0
        assert s.base


def test_prune_cmd_flags_per_arm():
    base, out = "/b", "/o"
    geom = runs.build_prune_cmd(_arm_spec("geometry"), base, out)
    assert "--no_band" in geom and "--no_purity" in geom
    assert "--random" in runs.build_prune_cmd(_arm_spec("random"), base, out)
    assert "--no_band" in runs.build_prune_cmd(_arm_spec("noband"), base, out)
    seg = runs.build_prune_cmd(_arm_spec("seg"), base, out)
    assert seg[seg.index("--allocation") + 1] == "per_segment"
    assert "--no_band" not in runs.build_prune_cmd(_arm_spec("semantic"), base, out)


def test_prune_cmd_shape_and_matched_budget():
    s = _arm_spec("semantic")
    cmd = runs.build_prune_cmd(s, "/base", "/out")
    assert cmd[1] == "prune_model.py"
    assert cmd[cmd.index("-m") + 1] == "/base"
    assert cmd[cmd.index("--out_dir") + 1] == "/out"
    assert cmd[cmd.index("--keep_frac") + 1] == str(s.keep_frac)
    # matched budget: every arm at the same (keep_frac, seed) keeps the same count
    assert len({s.keep_frac for s in runs.PRUNE_RUNS.values()}) == len(runs.PRUNE_KEEP_FRACS)


def test_prune_arms_cover_control_matrix():
    # the grid must contain the candidate, the control and the structure bar
    arms = {s.arm for s in runs.PRUNE_RUNS.values()}
    assert {"semantic", "geometry", "random"} <= arms
    # matched: for each keep_frac there is at least one of each core arm
    for kf in runs.PRUNE_KEEP_FRACS:
        for arm in ("semantic", "geometry", "random"):
            assert any(s.keep_frac == kf and s.arm == arm
                       for s in runs.PRUNE_RUNS.values())


def test_run_prune_dry():
    s = next(iter(runs.PRUNE_RUNS.values()))
    out = runs.run_prune(s, home_dir="/h", dry=True)
    assert str(out).replace("\\", "/").endswith(s.run_id)


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failures else 0)
