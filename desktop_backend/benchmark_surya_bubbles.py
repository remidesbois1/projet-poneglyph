"""Measure Surya through the desktop FastAPI endpoint on local, private fixtures."""

import argparse
import base64
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

from fastapi.testclient import TestClient

import local_ocr_server as server


def normalize(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'`":
        value = value[1:-1].strip()
    return re.sub(r"\s+", " ", value).strip()


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = json.loads(args.examples.read_text(encoding="utf-8"))
    items = [item for item in items if not str(item["id"]).startswith("stress")]
    items.sort(key=lambda item: (len(item["reference"]), str(item["id"])))
    if not 1 <= args.count <= len(items):
        raise ValueError("Requested count exceeds available examples")
    items = [items[round(i * (len(items) - 1) / max(1, args.count - 1))] for i in range(args.count)]

    with TestClient(server.app) as client:
        load = client.post("/model/load", params={"model_key": "surya"})
        load.raise_for_status()
        status = load.json()
        print(json.dumps({"model": status.get("artifact_revision"), "engine": status.get("generation_engine"),
                          "warmup_error": status.get("warmup_error"), "engine_error": status.get("optimized_engine_error")}), flush=True)
        rows = []
        for index, item in enumerate(items):
            image = args.examples.parent / item["image"]
            if not image.exists():
                image = args.examples.parent.parent / item["image"]
            payload = base64.b64encode(image.read_bytes()).decode("ascii")
            start = time.perf_counter()
            response = client.post("/ocr/text", params={"model_key": "surya"}, json={"image_bytes_base64": payload})
            response.raise_for_status()
            elapsed = 1000 * (time.perf_counter() - start)
            result = response.json()
            predicted = normalize(result["text"])
            target = normalize(item["reference"])
            native = normalize(item.get("native", item["reference"]))
            rows.append({"id": item["id"], "sha256": hashlib.sha256(predicted.encode()).hexdigest(),
                         "elapsed_ms": round(elapsed, 1), "edit_distance": distance(target, predicted),
                         "reference_chars": len(target), "exact": predicted == target,
                         "matches_native": predicted == native, "engine": result.get("generation_engine"),
                         "profile": result.get("generation_profile")})
            if (index + 1) % 8 == 0 or index + 1 == len(items):
                print(f"Measured {index + 1}/{len(items)}", flush=True)
    summary = {"sample_count": len(rows), "median_ms": statistics.median(r["elapsed_ms"] for r in rows),
               "mean_ms": statistics.mean(r["elapsed_ms"] for r in rows),
               "cer": sum(r["edit_distance"] for r in rows) / max(1, sum(r["reference_chars"] for r in rows)),
               "exact": sum(r["exact"] for r in rows), "matches_native": sum(r["matches_native"] for r in rows),
               "engines": sorted({str(r["engine"]) for r in rows})}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
