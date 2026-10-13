"""Audit the frozen Surya crop benchmark without exposing its transcriptions."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


def audit(directory):
    pages, ids, pixels = {}, set(), defaultdict(list)
    result = {"schema": 1, "splits": {}, "train_ids_excluded": []}
    for split in ("train", "val", "test"):
        path = directory / split / "metadata.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        split_pages, chars = set(), 0
        image_digest = hashlib.sha256()
        for row in rows:
            page, identity = str(row["page_id"]), str(row["id"])
            if page in pages and pages[page] != split:
                raise ValueError(f"Page crosses splits: {page}")
            if identity in ids:
                raise ValueError(f"Repeated annotation ID: {identity}")
            pages[page] = split
            ids.add(identity)
            split_pages.add(page)
            if not str(row["text"]).strip():
                raise ValueError(f"Empty target: {identity}")
            chars += len(str(row["text"]).strip())
            image_path = (directory / split / row["image_file"]).resolve()
            image_path.relative_to(directory.resolve())
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                if min(image.size) < 1:
                    raise ValueError(f"Empty crop: {identity}")
                digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            image_digest.update((identity + ":" + digest + "\n").encode())
            pixels[digest].append((split, identity))
        result["splits"][split] = {"samples": len(rows), "pages": len(split_pages), "reference_characters": chars,
            "metadata_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "ordered_pixels_sha256": image_digest.hexdigest()}
    overlaps = [items for items in pixels.values() if len({s for s, _ in items}) > 1]
    excluded = {identity for items in overlaps for split, identity in items if split == "train"}
    result["train_ids_excluded"] = sorted(excluded)
    result["cross_split_identical_crop_groups"] = len(overlaps)
    result["val_test_identical_crop_groups"] = sum({s for s, _ in items}.issuperset({"val", "test"}) for items in overlaps)
    result["source"] = "Frozen Poneglyph validated bubble crops used for the 2026-10-05 Surya benchmark"
    result["initialization"] = "Original LightOnOCR-3 0.8B; bbox weights rejected because of page overlap with this benchmark"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
