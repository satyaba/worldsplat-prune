"""Tests for the prompt-driven front-end's pure parts.

Currently covers :func:`worldsplat_prune.frontend.sam2_box.masks_to_class_maps`
(the per-class mask composition). No SAM2 / torch / transformers required —
only the pure composition is exercised; the detector/captioner/driver are added
alongside later.

    pytest tests/test_frontend.py
    python tests/test_frontend.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worldsplat_prune.frontend.sam2_box import masks_to_class_maps, _slug
from worldsplat_prune.frontend import grounding as g
from worldsplat_prune.frontend import caption as c


def test_masks_to_class_maps_are_independent_per_class():
    m_left = np.zeros((6, 6), dtype=bool)
    m_left[:, :3] = True
    m_right = np.zeros((6, 6), dtype=bool)
    m_right[:, 3:] = True
    out = masks_to_class_maps(
        {"chair": [m_left], "table": [m_right]},
        {"chair": [0.9], "table": [0.8]},
        fill_void=False)
    assert set(out) == {"chair", "table"}
    # ids restart at 1 in every class (re-based by class later)
    assert set(np.unique(out["chair"]).tolist()) == {0, 1}
    assert set(np.unique(out["table"]).tolist()) == {0, 1}
    assert out["chair"][0, 0] == 1 and out["chair"][0, 5] == 0
    assert out["table"][0, 5] == 1 and out["table"][0, 0] == 0


def test_masks_to_class_maps_score_ordered_overlap():
    a = np.zeros((4, 4), dtype=bool)
    a[:2, :] = True
    b = np.zeros((4, 4), dtype=bool)
    b[1:3, :] = True          # overlaps a on row 1
    out = masks_to_class_maps({"x": [a, b]}, {"x": [0.9, 0.5]}, fill_void=False)
    m = out["x"]
    assert int(m.max()) == 2          # both survive
    assert m[0, 0] == 1 and m[2, 0] == 2
    assert m[1, 0] == 1               # higher score keeps the contested row
    # if the scores are swapped, the other mask is painted first and owns row 1
    out2 = masks_to_class_maps({"x": [a, b]}, {"x": [0.5, 0.9]}, fill_void=False)
    assert out2["x"][0, 0] == 2       # a is now painted second
    assert out2["x"][2, 0] == 1       # b owns its uncontested rows


def test_masks_to_class_maps_min_area_and_empty_class():
    big = np.zeros((5, 5), dtype=bool)
    big[:2, :] = True                 # 10 px
    speck = np.zeros((5, 5), dtype=bool)
    speck[4, 4] = True                # 1 px
    out = masks_to_class_maps({"keep": [big, speck], "none": []},
                              min_area=3, fill_void=False)
    assert set(out) == {"keep"}       # empty class omitted
    assert set(np.unique(out["keep"]).tolist()) == {0, 1}


def test_slug_is_filesystem_safe():
    assert _slug("traffic light") == "traffic_light"
    assert _slug("cup/glass") == "cup_glass"
    assert _slug("  ") == "class"


# --------------------------------------------------------------------------- #
# grounding: phrase parsing / prompt / label matching                           #
# --------------------------------------------------------------------------- #
def test_parse_phrases_dedup_and_comments(tmpdir=None):
    import tempfile
    from pathlib import Path

    assert g.parse_phrases(["Chair.", "table", "chair"]) == ["chair", "table"]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "phrases.txt"
        p.write_text("chair\n# a comment\ntable .\nchair\n")
        assert g.parse_phrases(str(p)) == ["chair", "table"]


def test_format_text_prompt():
    assert g.format_text_prompt(["Chair.", " table"]) == "chair. table."


def test_match_label_to_phrase():
    phrases = ["traffic light", "chair"]
    assert g.match_label_to_phrase("traffic light", phrases) == "traffic light"
    assert g.match_label_to_phrase("Traffic Light.", phrases) == "traffic light"
    assert g.match_label_to_phrase("chair", phrases) == "chair"
    assert g.match_label_to_phrase("chair leg", phrases) == "chair"   # substring
    assert g.match_label_to_phrase("bicycle", phrases) is None


def test_extract_phrases_stopwords_and_bigrams():
    caps = c.extract_phrases("a brown wooden chair in a room")
    assert "chair" in caps and "room" in caps
    assert "a" not in caps and "in" not in caps          # stop-words dropped
    assert "wooden chair" in caps                        # adjacent bigram kept
    assert c.extract_phrases("the the the") == []


def test_merge_phrase_lists_order_preserving():
    assert c.merge_phrase_lists([["chair", "table"], ["table", "lamp"]]) == \
        ["chair", "table", "lamp"]


def test_seed_phrase_list_with_callable_captioner():
    import tempfile
    from pathlib import Path
    from PIL import Image

    with tempfile.TemporaryDirectory() as d:
        ds = Path(d)
        (ds / "images").mkdir()
        for i in range(3):
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
                ds / "images" / f"rgb_{i}.png")

        cap = c.CallableCaptioner(lambda rgb: "a brown wooden chair in a room")
        out = ds / "phrases_seed.txt"
        phrases = c.seed_phrase_list(ds, cap, n=2, out=out)

        assert "chair" in phrases and "wooden chair" in phrases
        assert out.exists() and "chair" in out.read_text()


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
