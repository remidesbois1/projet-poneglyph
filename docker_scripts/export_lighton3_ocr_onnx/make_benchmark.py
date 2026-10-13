"""Expose the frozen test locally as lossless RGB PNGs; never include in publication."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def main(dataset, native_report, output):
    rows = json.loads(native_report.read_text(encoding="utf-8"))["pages"]
    out = output / "fixtures/test"
    out.mkdir(parents=True, exist_ok=True)
    examples = []
    for index, row in enumerate(rows):
        source = dataset / "test" / row["image_file"]
        image = Image.open(source).convert("RGB")
        image.save(out / f"page-{row['page_id']}.png", compress_level=1)
        examples.append({"id": str(row["page_id"]), "split": "test",
            "image": f"fixtures/test/page-{row['page_id']}.png", "reference": row["reference"],
            "native": row["prediction"], "width": image.width, "height": image.height,
            "source_image_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
        if (index + 1) % 40 == 0:
            print(f"Prepared {index + 1}/{len(rows)} private pages", flush=True)
    (output / "fixtures/benchmark.json").write_text(json.dumps(examples, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared {len(examples)} frozen test pages for real browser scoring", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.dataset, args.native_report, args.output)
