"""Indra OCR sub-models applied to manga bubble crops.

Only det.onnx and rec.onnx (Latin) are used: the bubble is already isolated, the
text is French, and the target pipeline must stay light.  Pre/post-processing
comes from the official ``indra_ocr`` package (crop_line, pad_batch, decode,
page_to_input) so the models see exactly the inputs they were trained on.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from indra_ocr.det_post import decode, page_to_input
from indra_ocr.rec_preproc import crop_line, is_dark_line, pad_batch, width_batches
from indra_ocr.vocab import Vocab, canonical

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
INDRA_DIR = HERE / "cache/models-v1.1.0"
YOLO_ONNX = HERE.parent / "retrain_ppocr_pipeline/outputs/models-published/bubble_line_detector_yolo26n.onnx"
REC_OUTPUTS = ["ids", "conf", "lengths"]
LM_OUTPUTS = ["blank", "topv", "topi"]


def session(path, threads: int = 4) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_cpu_mem_arena = False  # as indra_ocr: varying shapes
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def load_rows(split: str) -> list[dict]:
    with open(OUT / f"ppocr_{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_gray(rel: str) -> np.ndarray:
    gray = cv2.imread(str(OUT / rel), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(rel)
    return gray


def boxes_array(boxes: list[dict]) -> np.ndarray:
    return np.array([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in boxes], np.float32).reshape(-1, 4)


def sort_line_boxes(boxes: np.ndarray) -> np.ndarray:
    """Production sortLineBoxes (ocr.worker.js): rows of half-pitch bands, then x."""
    if not len(boxes):
        return boxes
    heights = np.sort(boxes[:, 3] - boxes[:, 1])
    band = max(10.0, float(heights[len(heights) // 2]) * 0.65)
    # JS Math.round rounds .5 up; np.round is banker's rounding.
    keys = [(math.floor((b[1] + b[3]) / 2 / band + 0.5), b[0]) for b in boxes]
    return boxes[sorted(range(len(boxes)), key=lambda i: keys[i])]


def adjust_boxes(boxes: np.ndarray, grow_y: float, gray_shape) -> np.ndarray:
    """Grow (or shrink, negative) each box by grow_y x its height on top and bottom."""
    if not len(boxes) or not grow_y:
        return boxes
    out = boxes.copy()
    h = out[:, 3] - out[:, 1]
    out[:, 1] -= grow_y * h
    out[:, 3] += grow_y * h
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, gray_shape[0])
    return out


def whole_crop_box(gray: np.ndarray) -> np.ndarray:
    """Production fallback when no line is found: the bubble is one line."""
    return np.array([[0, 0, gray.shape[1], gray.shape[0]]], np.float32)


_HYPHEN_BREAK = re.compile(r"(\w)-$")


def join_lines(texts: list[str], dehyphenate: bool) -> str:
    out = ""
    for t in (t.strip() for t in texts):
        if not t:
            continue
        if not out:
            out = t
        elif dehyphenate and _HYPHEN_BREAK.search(out) and t[:1].islower():
            out = out[:-1] + t
        else:
            out += " " + t
    return out


class IndraDet:
    def __init__(self, threads: int = 4, path: Path = INDRA_DIR / "det.onnx"):
        self.sess = session(path, threads)

    def detect(self, gray: np.ndarray, scale: float = 1.0, margin: int = 0, thresh: float = 0.5) -> np.ndarray:
        """Line boxes [N, 4] in crop pixels.  scale: resize factor applied to the
        crop (a page goes in at long side 1024; a bubble crop is tiny, so its own
        resolution is the reference).  margin: white border added first, in crop px."""
        if margin:
            gray = cv2.copyMakeBorder(gray, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)
        long_side = max(32, round(max(gray.shape) * scale))
        x, s = page_to_input(gray, long_side)
        out = self.sess.run(None, {"x": x[None, None]})[0][0]
        boxes = decode(out[:3], thresh=thresh)[:, :4] / s
        if margin:
            boxes -= margin
        h, w = gray.shape[0] - 2 * margin, gray.shape[1] - 2 * margin
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        keep = (boxes[:, 2] - boxes[:, 0] >= 2) & (boxes[:, 3] - boxes[:, 1] >= 2)
        return boxes[keep]


class IndraRec:
    """rec.onnx (Latin).  Greedy CTC by default; ``lm`` re-reads unsure lines with
    the official n-gram beam search (as indra_ocr does for lines under the gate)."""

    def __init__(self, threads: int = 4, path: Path = INDRA_DIR / "rec.onnx", lm: bool = False):
        meta = json.loads((INDRA_DIR / "meta.json").read_text(encoding="utf-8"))
        self.vocab = Vocab.from_json(meta["vocab"])
        self.height = meta["rec_height"]
        self.window = meta.get("rec_window", 1)
        self.sess = session(path, threads)
        self.lm = None
        if lm:
            from indra_ocr.lm import NgramLM

            self.lm = NgramLM(INDRA_DIR / "lm.npz")

    def crops(self, gray: np.ndarray, boxes: np.ndarray) -> list[np.ndarray]:
        return [crop_line(gray, b, self.height, window=self.window, dark=is_dark_line(gray, b)) for b in boxes]

    def _run(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        out: list = [("", 0.0)] * len(crops)
        for part in width_batches([c.shape[1] for c in crops]):
            arr, widths = pad_batch([crops[i] for i in part])
            names = REC_OUTPUTS + (LM_OUTPUTS if self.lm is not None else [])
            res = self.sess.run(names, {"x": arr, "widths": widths})
            for k, i in enumerate(part):
                n = int(res[2][k])
                seq, pm = res[0][k, :n], res[1][k, :n]
                keep = (seq != 0) & np.concatenate([[True], seq[1:] != seq[:-1]])
                ids = seq[keep].tolist()
                conf = float(pm[keep].mean()) if keep.any() else 0.0
                if self.lm is not None and 0.5 <= conf < 0.9:
                    from indra_ocr.lm import beam_search

                    ids, _ = beam_search(res[3][k, :n], res[4][k, :n], res[5][k, :n], self.lm, 0.5, 1.5, 8, cand_floor=3e-3)
                out[i] = (self.vocab.decode(ids), conf)
        return out

    def read_lines(self, gray: np.ndarray, boxes: np.ndarray) -> list[tuple[str, float]]:
        return self._run(self.crops(gray, boxes))

    def stitch(self, gray: np.ndarray, boxes: np.ndarray, gap: int) -> np.ndarray:
        """Line windows of one bubble side by side, in reading order: one input."""
        crops = self.crops(gray, boxes)
        if gap and len(crops) > 1:
            pad = np.zeros((crops[0].shape[0], gap), np.float32)
            crops = [c for crop in crops for c in (crop, pad)][:-1]
        return np.concatenate(crops, axis=1)

    def read_stitched(self, gray: np.ndarray, boxes: np.ndarray, gap: int) -> tuple[str, float]:
        return self._run([self.stitch(gray, boxes, gap)])[0]


class YoloLines:
    """Python port of the production YOLO line detector, for the speed benchmark
    only (accuracy uses the boxes produced by the production worker)."""

    def __init__(self, threads: int = 4, conf: float = 0.10, nms_iou: float = 0.75):
        self.sess = session(YOLO_ONNX, threads)
        shape = self.sess.get_inputs()[0].shape
        self.h = shape[2] if isinstance(shape[2], int) else 1024
        self.w = shape[3] if isinstance(shape[3], int) else 1024
        self.conf, self.nms_iou = conf, nms_iou

    def detect(self, bgr: np.ndarray) -> np.ndarray:
        ih, iw = bgr.shape[:2]
        scale = min(self.w / iw, self.h / ih)
        nw, nh = round(iw * scale), round(ih * scale)
        px, py = (self.w - nw) / 2, (self.h - nh) / 2
        canvas = np.full((self.h, self.w, 3), 128, np.uint8)
        canvas[int(py): int(py) + nh, int(px): int(px) + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.sess.get_inputs()[0].name: x})[0].reshape(-1, 6)
        out = out[out[:, 4] >= self.conf]
        b = np.stack([(out[:, 0] - px) / scale, (out[:, 1] - py) / scale, (out[:, 2] - px) / scale, (out[:, 3] - py) / scale], 1)
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, iw)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, ih)
        ok = ((b[:, 2] - b[:, 0]) >= 4) & ((b[:, 3] - b[:, 1]) >= 4) & ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) >= 24)
        b, score = b[ok], out[ok, 4]
        kept: list[int] = []
        for i in np.argsort(-score):
            if all(iou(b[i], b[j]) < self.nms_iou for j in kept):
                kept.append(int(i))
        return sort_line_boxes(b[kept].astype(np.float32))


class PpocrRec:
    """Python port of the production stitch + PP-OCRv6 recognizer, for the speed
    benchmark only."""

    def __init__(self, threads: int = 4):
        models = YOLO_ONNX.parent
        self.sess = session(models / "ppocrv6_bubble_line_rec_webgpu.onnx", threads)
        m = json.loads((models / "browser_manifest.json").read_text(encoding="utf-8"))
        self.chars = m["character_list"]
        self.height, self.min_w, self.max_w = m.get("image_height", 48), m.get("min_image_width", 960), m.get("max_image_width", 3200)

    def read(self, bgr: np.ndarray, boxes: np.ndarray, pad: int = 2, gap: int = 8) -> str:
        H, W = bgr.shape[:2]
        boxes = boxes if len(boxes) else whole_crop_box(bgr[:, :, 0])
        crops = []
        for x1, y1, x2, y2 in boxes:
            a, b = max(0, math.floor(x1) - pad), max(0, math.floor(y1) - pad)
            c, d = min(W, math.ceil(x2) + pad), min(H, math.ceil(y2) + pad)
            crops.append(bgr[b:max(d, b + 1), a:max(c, a + 1)])
        th = max(c.shape[0] for c in crops)
        parts = []
        for c in crops:
            if c.shape[0] != th:
                c = cv2.resize(c, (max(1, round(c.shape[1] * th / c.shape[0])), th), interpolation=cv2.INTER_LINEAR)
            parts += [c, np.full((th, gap, 3), 255, np.uint8)]
        stitched = np.concatenate(parts[:-1], 1)
        aw = math.ceil(self.height * stitched.shape[1] / stitched.shape[0])
        width = min(self.max_w, max(self.min_w, aw))
        img = np.full((self.height, width, 3), 128, np.uint8)
        img[:, : min(width, aw)] = cv2.resize(stitched, (min(width, aw), self.height), interpolation=cv2.INTER_LINEAR)
        x = ((img.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]  # BGR, as production
        out = self.sess.run(None, {self.sess.get_inputs()[0].name: x})[0][0]
        best = out.argmax(-1)
        text, prev = [], -1
        for k in best:
            if k != 0 and k != prev and k < len(self.chars):
                text.append(self.chars[k])
            prev = k
        return "".join(text)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-6)


def match_rate(ref: np.ndarray, pred: np.ndarray, thr: float = 0.5) -> tuple[int, int, int]:
    """Greedy one-to-one IoU matches: (matched, len(ref), len(pred))."""
    pairs = sorted(((iou(r, p), i, j) for i, r in enumerate(ref) for j, p in enumerate(pred)), reverse=True)
    used_r, used_p, m = set(), set(), 0
    for v, i, j in pairs:
        if v < thr:
            break
        if i in used_r or j in used_p:
            continue
        used_r.add(i)
        used_p.add(j)
        m += 1
    return m, len(ref), len(pred)


class Timer:
    def __init__(self):
        self.ms: dict[str, float] = {}

    def __call__(self, key: str):
        timer = self

        class _T:
            def __enter__(self):
                self.t = time.perf_counter()

            def __exit__(self, *exc):
                timer.ms[key] = timer.ms.get(key, 0.0) + (time.perf_counter() - self.t) * 1000

        return _T()


def oov(vocab: Vocab, text: str) -> set[str]:
    return vocab.oov(canonical(" ".join(text.split())))
