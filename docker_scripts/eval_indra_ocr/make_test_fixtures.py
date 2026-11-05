"""Reference outputs of OpenCV / indra_ocr for frontend/src/lib/indraOcr.test.js.

  python make_test_fixtures.py   -> frontend/src/lib/indraOcr.fixtures.json
"""

from __future__ import annotations

import json

import cv2
import numpy as np
from indra_ocr.det_post import decode, page_to_input
from indra_ocr.rec_preproc import crop_line, is_dark_line

from indra_lib import HERE

rng = np.random.default_rng(7)


def image(h, w):
    return cv2.GaussianBlur(rng.integers(0, 256, (h, w), dtype=np.uint8), (0, 0), 1.2)


resize = []
for (sh, sw, dh, dw, mode) in [(37, 53, 20, 29, "area"), (64, 48, 16, 12, "area"), (30, 41, 29, 40, "area"),
                               (12, 17, 31, 44, "linear"), (9, 25, 27, 50, "linear"), (20, 20, 21, 33, "linear")]:
    src = image(sh, sw)
    out = cv2.resize(src, (dw, dh), interpolation=cv2.INTER_AREA if mode == "area" else cv2.INTER_LINEAR)
    resize.append({"mode": mode, "sw": sw, "sh": sh, "dw": dw, "dh": dh, "src": src.ravel().tolist(), "out": out.ravel().tolist()})

# A bubble-like page: white ground with a dark-text line, and a light-on-dark band.
page = np.full((70, 90), 245, np.uint8)
page[11:21, 8:80:4] = 30  # dark strokes on light paper
page[11:13, 8:80] = 30
page[40:56, 5:85] = 20
page[44:52, 15:70] = 235
page = cv2.GaussianBlur(page, (0, 0), 0.8)
x, s = page_to_input(page, 68)
crops = []
for box in ([7.5, 9.2, 80.4, 22.6], [4.0, 39.5, 86.0, 56.2]):
    ink = crop_line(page, box, 32, window=3)
    # Ink values are k/255 by construction: store k.
    crops.append({"box": box, "dark": bool(is_dark_line(page, box)), "width": ink.shape[1], "height": ink.shape[0],
                  "levels": np.round(ink * 255).astype(int).ravel().tolist()})

# Detector output maps: two cores (and one single pixel, dropped) with top/bottom distances.
h, w = 12, 16
out = np.full((3, h, w), -4.0, np.float32)
out[0, 2:4, 3:11] = 3.0
out[1, 2:4, 3:11] = 5.0
out[2, 2:4, 3:11] = 6.0
out[0, 8, 5] = 2.0
out[0, 7:10, 12:15] = 1.0
out[1, 7:10, 12:15] = np.linspace(2, 4, 9, dtype=np.float32).reshape(3, 3)
out[2, 7:10, 12:15] = 3.5
boxes = decode(out)

fixtures = {
    "generator": "docker_scripts/eval_indra_ocr/make_test_fixtures.py (OpenCV %s, indra_ocr 1.1.2)" % cv2.__version__,
    "resize": resize,
    "page": {"width": 90, "height": 70, "gray": page.ravel().tolist(), "long_side": 68,
             "input": {"width": x.shape[1], "height": x.shape[0], "scale": s, "levels": np.round(x * 255).astype(int).ravel().tolist()},
             "crops": crops},
    "decode": {"h": h, "w": w, "out": out.ravel().tolist(), "boxes": np.round(boxes, 4).tolist()},
}
path = HERE.parents[1] / "frontend/src/lib/indraOcr.fixtures.json"
path.write_text(json.dumps(fixtures, separators=(",", ":")), encoding="utf-8")
print(path, path.stat().st_size)
