"""Private frozen bubble test, lossless images and native reference predictions."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image


def main(dataset, native_report, output):
    report = json.loads(native_report.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (dataset / "test/metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    native = {str(row["id"]): row for row in report["samples"]}
    if len(native) != 1612 or {str(row["id"]) for row in rows} != set(native):
        raise ValueError("Native report and frozen test do not contain the same 1612 crops")
    out = output / "fixtures/test"
    out.mkdir(parents=True, exist_ok=True)
    examples = []
    for row in rows:
        reference = native[str(row["id"])]
        if reference["reference"] != row["text"]:
            raise ValueError("Test reference changed since native evaluation")
        source = dataset / "test" / row["image_file"]
        image = Image.open(source).convert("RGB")
        image.save(out / f"bubble-{row['id']}.png", compress_level=1)
        examples.append({"id": str(row["id"]), "page_id": str(row["page_id"]), "split": "test",
            "image": f"fixtures/test/bubble-{row['id']}.png", "reference": row["text"],
            "native": reference["prediction"], "width": image.width, "height": image.height,
            "source_image_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    (output / "fixtures/benchmark.json").write_text(json.dumps(examples, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared {len(examples)} private frozen crops; no images or predictions in publication")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.dataset, args.native_report, args.output)
