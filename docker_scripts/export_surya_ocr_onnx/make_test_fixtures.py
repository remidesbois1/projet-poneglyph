"""Replay an existing full native benchmark in the browser, without publishing crops."""
import argparse
import json
import shutil
from pathlib import Path


def main(dataset, benchmark, output):
    samples = {str(item["id"]): item for item in json.loads(benchmark.read_text(encoding="utf-8"))["samples"]}
    rows = [json.loads(line) for line in (dataset / "test/metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    if set(samples) != {str(row["id"]) for row in rows}:
        raise ValueError("Benchmark and test dataset do not contain the same crop IDs")
    output.mkdir(parents=True, exist_ok=True)
    examples = []
    for index, row in enumerate(sorted(rows, key=lambda r: len(r["text"]))):
        native = samples[str(row["id"])]
        if native["reference"] != row["text"]:
            raise ValueError(f"Reference changed for {row['id']}")
        filename = f"bubble-{index}.png"
        shutil.copy2(dataset / "test" / row["image_file"], output / filename)
        examples.append({"id": str(row["id"]), "image": filename, "reference": row["text"], "native": native["prediction"]})
    (output / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(examples)} private test fixtures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.dataset, args.benchmark, args.output)
