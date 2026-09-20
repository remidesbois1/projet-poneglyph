import io
import json
from pathlib import Path
import re
import shutil
import zipfile

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

CLASS_NAMES = ("cover", "story_page", "annexe", "summary")
MODEL_IMAGE_SIZE = 224
MODEL_RESIZE_SIZE = 256
MODEL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
MODEL_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BASE_DIR = Path(__file__).resolve().parent
CBZ_DIR = BASE_DIR.parents[1] / "scripts" / "One Piece Tomes"
OUTPUT_DATASET_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "runs" / "final-20260718T154451Z" / "page_type_classifier.onnx"


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value)]


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    scale = MODEL_RESIZE_SIZE / min(w, h)
    resized = image.resize((round(w * scale), round(h * scale)), Image.Resampling.BILINEAR)
    left = (resized.width - MODEL_IMAGE_SIZE) // 2
    top = (resized.height - MODEL_IMAGE_SIZE) // 2
    cropped = resized.crop((left, top, left + MODEL_IMAGE_SIZE, top + MODEL_IMAGE_SIZE))
    arr = np.asarray(cropped, dtype=np.float32) / 255.0
    norm = (arr - MODEL_MEAN) / MODEL_STD
    tensor = norm.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor, dtype=np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def main():
    if OUTPUT_DATASET_DIR.exists():
        shutil.rmtree(OUTPUT_DATASET_DIR)
    OUTPUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Chargement du modèle {MODEL_PATH.name}...")
    session = ort.InferenceSession(str(MODEL_PATH))
    input_name = session.get_inputs()[0].name

    manifest_volumes = []
    labels_dict = {}

    total_images_processed = 0

    for tome_num in range(1, 8):
        cbz_filename = f"One Piece T{tome_num:02d}.cbz"
        cbz_path = CBZ_DIR / cbz_filename
        if not cbz_path.exists():
            print(f"Archive manquante : {cbz_path}")
            continue

        volume_id = f"one-piece-t{tome_num:02d}"
        volume_display = f"One Piece T{tome_num:02d}"

        manifest_volumes.append({
            "id": volume_id,
            "display_name": volume_display,
            "cbz_file": cbz_filename,
        })

        volume_images_dir = OUTPUT_DATASET_DIR / "volumes" / volume_id / "images"
        volume_images_dir.mkdir(parents=True, exist_ok=True)

        print(f"Extraction & inférence pour {volume_display}...")
        with zipfile.ZipFile(cbz_path) as z:
            names = [
                n for n in z.namelist()
                if not n.endswith("/")
                and not n.startswith("__MACOSX")
                and any(n.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"])
            ]
            names.sort(key=natural_key)

            for idx, internal_path in enumerate(names, start=1):
                page_filename = f"{idx:04d}.jpg"
                out_path = volume_images_dir / page_filename

                raw_bytes = z.read(internal_path)
                img = Image.open(io.BytesIO(raw_bytes))
                img = ImageOps.exif_transpose(img).convert("RGB")
                img.save(out_path, "JPEG", quality=95)

                tensor = preprocess_image(img)
                logits = session.run(None, {input_name: tensor})[0][0]
                probs = softmax(logits).tolist()

                class_probs = {cls_name: float(probs[i]) for i, cls_name in enumerate(CLASS_NAMES)}
                best_idx = int(np.argmax(probs))
                predicted_class = CLASS_NAMES[best_idx]
                confidence = float(probs[best_idx])

                page_id = f"{volume_id}:{idx:04d}"
                orig_file_name = internal_path.split("/")[-1]

                labels_dict[page_id] = {
                    "volume_id": volume_id,
                    "volume_name": volume_display,
                    "page_file": page_filename,
                    "original_file": orig_file_name,
                    "page_num": idx,
                    "label": predicted_class,
                    "predicted_label": predicted_class,
                    "confidence": confidence,
                    "probabilities": class_probs,
                    "status": "auto_predicted",
                }
                total_images_processed += 1

        print(f"  {volume_display} terminé ({len(names)} pages).")

    manifest_data = {"volumes": manifest_volumes}
    (OUTPUT_DATASET_DIR / "manifest.json").write_text(
        json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    labels_data = {"labels": labels_dict}
    (OUTPUT_DATASET_DIR / "labels.json").write_text(
        json.dumps(labels_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    counts = {}
    for item in labels_dict.values():
        c = item["label"]
        counts[c] = counts.get(c, 0) + 1

    print("Extraction et inférence terminées !")
    print(f"Total pages extraites : {total_images_processed}")
    print(f"Répartition des prédictions : {counts}")


if __name__ == "__main__":
    main()
