"""Reuse the validated, page-split Poneglyph export, preserving native grounding."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image

from grounding import PROMPT, format_grounding, parse_grounding


def prepare(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    seen_pages, seen_images = {}, {}
    report = {"source": str(source), "prompt": PROMPT, "thinking": False, "bbox_scale": 1000, "splits": {}}
    for split in ("train", "val", "test"):
        metadata = source / split / "metadata.jsonl"
        records = [json.loads(line) for line in metadata.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            raise ValueError(f"Empty split {split}")
        target = destination / split
        (target / "images").mkdir(parents=True, exist_ok=True)
        output, hashes = [], []
        for row in records:
            page_id = row["page_id"]
            if page_id in seen_pages:
                raise ValueError(f"Duplicate or leaking page {page_id}: {seen_pages[page_id]}, {split}")
            seen_pages[page_id] = split
            items = row.get("bubbles")
            if not items:
                raise ValueError("Use the Surya full-page export with explicit bubbles and reading order")
            if any(item.get("order") is None for item in items):
                raise ValueError(f"Missing reading order on page {page_id}")
            items = sorted(items, key=lambda item: (int(item["order"]), item["id"]))
            response = format_grounding(items)
            parsed, invalid = parse_grounding(response)
            if invalid or len(parsed) != len(items):
                raise ValueError(f"Grounding roundtrip failed on {page_id}")
            image_path = source / split / row["image_file"]
            with Image.open(image_path) as image:
                image.load()
                size = list(image.size)
                digest = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
            if digest in seen_images and seen_images[digest] != split:
                raise ValueError(f"Identical image leaks between {seen_images[digest]} and {split}")
            seen_images[digest] = split
            image_target = target / row["image_file"]
            image_target.parent.mkdir(parents=True, exist_ok=True)
            # A frozen export must not share writable image inodes with its source.
            if image_target.exists():
                image_target.unlink()
            shutil.copy2(image_path, image_target)
            output.append({"id": page_id, "page_id": page_id, "split": split, "image_file": row["image_file"],
                           "image_size": size, "num_bubbles": len(items), "text": response, "bubbles": items,
                           "image_sha256": digest, "prompt": PROMPT})
            hashes.append(digest)
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output)
        (target / "metadata.jsonl").write_text(text, encoding="utf-8")
        report["splits"][split] = {"pages": len(output), "bubbles": sum(row["num_bubbles"] for row in output),
                                   "metadata_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                   "image_manifest_sha256": hashlib.sha256("\n".join(hashes).encode()).hexdigest(),
                                   "max_longest_edge": max(max(row["image_size"]) for row in output)}
    (destination / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path(__file__).parent / "dataset")
    args = parser.parse_args()
    prepare(args.source, args.destination)
