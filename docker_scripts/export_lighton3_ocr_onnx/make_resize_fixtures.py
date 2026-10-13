"""Private uint8 pixel goldens from the exact native Torchvision processor."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import InterpolationMode
from transformers import AutoProcessor


def main(source, directory, unit_output=None):
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    if processor.image_processor.backend != "torchvision":
        raise ValueError("These golden pixels require the native training processor's Torchvision backend")
    fixtures = json.loads((directory / "fixtures/examples.json").read_text(encoding="utf-8"))
    out = directory / "fixtures/resize"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    inputs = [(Image.open(directory / r["image"]).convert("RGB"), r["grid"][2] * 16, r["grid"][1] * 16, r["id"]) for r in fixtures]
    for index, (w, h, ow, oh) in enumerate([(17, 19, 32, 64), (71, 37, 32, 16), (257, 111, 128, 224),
                                           (32, 16, 32, 64), (32, 64, 64, 64), (64, 64, 64, 64)]):
        inputs.append((Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)), ow, oh, f"synthetic-{index}"))
    examples = []
    for index, (image, width, height, identity) in enumerate(inputs):
        pixels = np.asarray(image)
        tensor = torch.from_numpy(pixels.copy()).permute(2, 0, 1)
        resized = F.resize(tensor, [height, width], interpolation=InterpolationMode.BICUBIC, antialias=True)
        src = np.concatenate([pixels, np.full((*pixels.shape[:2], 1), 255, dtype=np.uint8)], axis=2)
        dest = resized.permute(1, 2, 0).numpy()
        dest = np.concatenate([dest, np.full((*dest.shape[:2], 1), 255, dtype=np.uint8)], axis=2)
        src.tofile(out / f"input-{index}.bin")
        dest.tofile(out / f"output-{index}.bin")
        examples.append({"id": identity, "width": image.width, "height": image.height,
                         "output_width": width, "output_height": height, "input": f"input-{index}.bin", "output": f"output-{index}.bin"})
    (out / "examples.json").write_text(json.dumps(examples, indent=2))
    if unit_output:
        golden = []
        for w, h, ow, oh in [(3, 2, 5, 4), (9, 7, 4, 3), (3, 2, 3, 4), (3, 2, 6, 2)]:
            pixels = ((np.arange(w * h * 3).reshape(h, w, 3) * 37 + 11) % 256).astype(np.uint8)
            expected = F.resize(torch.from_numpy(pixels.copy()).permute(2, 0, 1), [oh, ow],
                                interpolation=InterpolationMode.BICUBIC, antialias=True).permute(1, 2, 0).numpy()
            rgba = lambda a: np.concatenate([a, np.full((*a.shape[:2], 1), 255, dtype=np.uint8)], axis=2).reshape(-1).tolist()
            golden.append({"width": w, "height": h, "outputWidth": ow, "outputHeight": oh,
                           "input": rgba(pixels), "expected": rgba(expected)})
        unit_output.write_text(json.dumps({"source": "PyTorch 2.8 / Torchvision uint8 CPU BICUBIC antialias=True",
                                           "cases": golden}, separators=(",", ":")) + "\n")
    print(f"Saved {len(examples)} native uint8 resize goldens, backend={processor.image_processor.backend}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--unit-output", type=Path)
    args = parser.parse_args()
    main(args.source, args.directory, args.unit_output)
