import importlib.util
import io
import os
from pathlib import Path

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-key")


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = load_module("bbox_train_test_module", "train_lighton_bbox.py")
export = load_module("bbox_export_test_module", "export_dataset.py")


def result(reference, prediction):
    return {
        "page_id": 1,
        "gt_text": reference,
        "pred_text": prediction,
        "gt_items": train.parse_bbox_output(reference),
        "pred_items": train.parse_bbox_output(prediction),
        "inference_time": 1.0,
        "error": None,
    }


def test_strict_training_prompt_contract():
    assert "Texte exact [x1,y1,x2,y2]" in train.OUTPUT_CONTRACT
    assert "N'ajoute aucun JSON" in train.OUTPUT_CONTRACT
    messages = train.messages_for_entry(Path("unused.png"), None)
    assert messages == [{"role": "user", "content": [{"type": "image"}]}]


def test_benchmark_metadata_keeps_image_only_conditioning(tmp_path):
    path = tmp_path / "benchmark.json"
    train.save_benchmark(path, "model", {"cer": 0.1}, [])
    payload = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert payload["conditioning"] == "image_only"
    assert payload["prompt"] == ""
    assert payload["output_contract"] == train.OUTPUT_CONTRACT


def test_bbox_parser_rejects_invalid_coordinates_and_lines():
    parsed = train.parse_bbox_output(
        "Bonjour [10,20,900,200]\ninvalid\nOutside [4,4,1100,8]\nDegenerate [4,4,4,8]"
    )
    assert parsed == [{"text": "Bonjour", "bbox": [10, 20, 900, 200]}]


def test_perfect_page_metrics_are_perfect():
    text = "Salut ! [10,20,100,120]\nCa va ? [500,400,700,600]"
    metrics = train.compute_metrics_from_results([result(text, text)])
    assert metrics["cer"] == 0
    assert metrics["f1@0_5"] == 1
    assert metrics["mean_iou"] == 1
    assert metrics["avg_detection_rate"] == 1


def test_missing_and_extra_boxes_are_penalized():
    reference = "A [0,0,100,100]\nB [200,200,300,300]"
    prediction = "A [0,0,100,100]\nC [700,700,800,800]"
    metrics = train.compute_metrics_from_results([result(reference, prediction)])
    assert metrics["f1@0_5"] == 0.5
    assert metrics["avg_detection_rate"] == 0.5


def test_resize_never_exceeds_1500_and_preserves_small_pages():
    large = Image.new("RGB", (3000, 2000))
    resized, width, height = export.resize_page(large, export.TARGET_LONGEST_SIDE)
    assert resized.size == (1500, 1000)
    assert (width, height) == resized.size
    small = Image.new("RGB", (900, 1200))
    untouched, width, height = export.resize_page(small, export.TARGET_LONGEST_SIDE)
    assert untouched.size == (900, 1200)
    assert (width, height) == (900, 1200)


def test_frozen_splits_keep_duplicates_together(tmp_path):
    manifest_path = tmp_path / "split_manifest.json"
    splits, _ = export.build_frozen_splits(
        {1: "same", 2: "same", 3: "three", 4: "four", 5: "five"},
        manifest_path,
    )
    split_by_page = {
        page_id: split for split, page_ids in splits.items() for page_id in page_ids
    }
    assert split_by_page[1] == split_by_page[2]
    frozen_test = set(splits["test"])
    updated, _ = export.build_frozen_splits(
        {1: "same", 2: "same", 3: "three", 4: "four", 5: "five", 6: "six"},
        manifest_path,
    )
    assert set(updated["test"]) == frozen_test


def test_empty_failed_manifest_is_reinitialized(tmp_path):
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(
        '{"version":2,"splits":{"train":[],"val":[],"test":[]},"page_hashes":{}}',
        encoding="utf-8",
    )
    splits, _ = export.build_frozen_splits(
        {index: f"hash-{index}" for index in range(1, 21)},
        manifest_path,
    )
    assert splits["train"]
    assert splits["val"]
    assert splits["test"]


def test_partial_manifest_with_empty_test_is_reinitialized(tmp_path):
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(
        '{"version":2,"splits":{"train":[1,2,3,4],"val":[5],"test":[]},'
        '"page_hashes":{"1":"a","2":"b","3":"c","4":"d","5":"e"}}',
        encoding="utf-8",
    )
    splits, _ = export.build_frozen_splits(
        {index: f"hash-{index}" for index in range(1, 21)},
        manifest_path,
    )
    assert splits["train"]
    assert splits["val"]
    assert splits["test"]


def test_private_r2_reference_contract():
    assert export.parse_r2_reference(
        "r2://poneglyph-pages-private/tome-27/page%201.avif",
        "poneglyph-pages-private",
    ) == ("poneglyph-pages-private", "tome-27/page 1.avif")


@pytest.mark.parametrize(
    "reference",
    [
        "r2://other-bucket/page.avif",
        "r2://poneglyph-pages-private/../page.avif",
        "r2://poneglyph-pages-private/foo%2Fbar.avif",
        "r2://poneglyph-pages-private/page.avif?download=1",
        " r2://poneglyph-pages-private/page.avif",
    ],
)
def test_rejects_noncanonical_or_wrong_bucket_r2_reference(reference):
    with pytest.raises(ValueError):
        export.parse_r2_reference(reference, "poneglyph-pages-private")


def _jpeg_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buffer, "JPEG")
    return buffer.getvalue()


def test_download_pages_routes_private_r2_through_s3(monkeypatch):
    content = _jpeg_bytes()

    class Body:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return content

    class Client:
        def __init__(self):
            self.calls = []

        def get_object(self, *, Bucket, Key):
            self.calls.append((Bucket, Key))
            return {"Body": Body()}

    client = Client()
    monkeypatch.setenv("R2_PAGES_BUCKET_NAME", "poneglyph-pages-private")
    monkeypatch.setattr(export, "create_r2_client", lambda: client)
    pages = {
        42: {
            "url_image": "r2://poneglyph-pages-private/tome-27/page%201.avif"
        }
    }
    downloaded = export.download_pages(pages)
    assert downloaded[42].size == (16, 16)
    assert client.calls == [("poneglyph-pages-private", "tome-27/page 1.avif")]


def test_download_pages_refuses_partial_dataset(monkeypatch):
    content = _jpeg_bytes()

    class Response:
        def __init__(self, payload):
            self.content = payload

        def raise_for_status(self):
            return None

    def fake_get(url, timeout):
        assert timeout == export.REQUEST_TIMEOUT_SECONDS
        if url.endswith("bad.jpg"):
            raise OSError("synthetic download failure")
        return Response(content)

    monkeypatch.setattr(export.requests, "get", fake_get)
    pages = {
        1: {"url_image": "https://example.test/good.jpg"},
        2: {"url_image": "https://example.test/bad.jpg"},
    }
    with pytest.raises(RuntimeError, match="refusing to create a partial bbox dataset"):
        export.download_pages(pages)


def test_optimizer_step_estimate_respects_physical_batch():
    assert train.estimate_total_update_steps(100, 10, 1, 3) == 30
    assert train.estimate_total_update_steps(100, 5, 2, 3) == 30
    assert train.estimate_total_update_steps(100, 10, 1, 3, smoke_steps=7) == 7


def test_large_default_lora_profile(monkeypatch):
    monkeypatch.delenv("LIGHTON_BBOX_LORA_R", raising=False)
    monkeypatch.delenv("LIGHTON_BBOX_LORA_ALPHA", raising=False)
    monkeypatch.delenv("LIGHTON_BBOX_MODULES_TO_SAVE", raising=False)
    config = train.build_lora_config()
    assert config.r == 128
    assert config.lora_alpha == 256
    assert "vision_projection" in config.modules_to_save


def test_calibration_projection_skips_unsafe_next_batch():
    total = 32 * 2**30
    baseline = 5 * 2**30
    observed_peak = int(25.87 * 2**30)
    projected = train._project_next_calibration_ratio(
        baseline_bytes=baseline,
        observed_peak_bytes=observed_peak,
        observed_batch=1,
        next_batch=2,
        total_vram=total,
    )
    assert projected > 1.0
