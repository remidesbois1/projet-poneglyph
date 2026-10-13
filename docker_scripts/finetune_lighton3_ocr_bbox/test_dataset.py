import json

import pytest
from PIL import Image

from prepare_dataset import prepare


def source_dataset(root):
    for index, split in enumerate(("train", "val", "test")):
        directory = root / split
        (directory / "images").mkdir(parents=True)
        Image.new("RGB", (64, 96), (20 + index * 50, 20, 20)).save(directory / "images" / "page.png")
        row = {"page_id": index, "image_file": "images/page.png",
               "bubbles": [{"id": index, "order": 1, "text": "Bonjour !", "bbox": [100, 100, 600, 600]}]}
        (directory / "metadata.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_export_is_frozen_when_source_image_changes(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source_dataset(source)
    prepare(source, target)
    frozen = (target / "train" / "images" / "page.png").read_bytes()
    Image.new("RGB", (64, 96), "blue").save(source / "train" / "images" / "page.png")
    assert (target / "train" / "images" / "page.png").read_bytes() == frozen


def test_identical_content_across_splits_is_rejected(tmp_path):
    source_dataset(tmp_path / "source")
    source = tmp_path / "source"
    (source / "val" / "images" / "page.png").write_bytes((source / "train" / "images" / "page.png").read_bytes())
    with pytest.raises(ValueError, match="Identical image leaks"):
        prepare(source, tmp_path / "target")


def test_missing_reading_order_is_rejected(tmp_path):
    source_dataset(tmp_path / "source")
    path = tmp_path / "source" / "train" / "metadata.jsonl"
    row = json.loads(path.read_text())
    row["bubbles"][0]["order"] = None
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Missing reading order"):
        prepare(tmp_path / "source", tmp_path / "target")
