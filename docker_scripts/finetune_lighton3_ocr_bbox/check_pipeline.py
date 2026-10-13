"""Real processor checks for grounding, disabled thinking and assistant-only loss."""

import tempfile
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor

from grounding import format_grounding
from train import Collator, MODEL_ID, REVISION


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    processor.tokenizer.padding_side = "left"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "train" / "images").mkdir(parents=True)
        image = Image.new("RGB", (512, 768), "white")
        image.save(root / "train" / "images" / "page.jpg")
        image.close()
        rows = [{"page_id": i, "split": "train", "image_file": "images/page.jpg",
                 "text": format_grounding([{"bbox": [50, 50, 500, 200], "text": text}])}
                for i, text in enumerate(("Écoute !", "Un texte nettement plus long pour contrôler la longueur variable."))]
        batch = Collator(processor, root, augment=False, cache_items=1)(rows)
        for i, row in enumerate(rows):
            supervised = batch["labels"][i][batch["labels"][i].ne(-100)].tolist()
            decoded = processor.tokenizer.decode(supervised, skip_special_tokens=False)
            assert decoded.startswith(row["text"]), decoded
            assert decoded.endswith("<|im_end|>\n"), repr(decoded)
            assert "<think>" not in decoded and "grounding" not in decoded
            assert batch["labels"][i][batch["attention_mask"][i].eq(0)].eq(-100).all()
        assert batch["ocr_targets"].numel() == batch["labels"][:, 1:].ne(-100).sum()
        for pos, token in zip(batch["ocr_positions"].tolist(), batch["ocr_targets"].tolist()):
            row, col = divmod(pos, batch["labels"].shape[1])
            assert batch["input_ids"][row, col + 1].item() == token
        weighted = Collator(processor, root, augment=False, cache_items=1, bbox_loss_weight=.2, eos_loss_weight=2)(rows)
        assert weighted["ocr_targets"].tolist() == batch["ocr_targets"].tolist()
        assert set(round(float(v), 2) for v in weighted["ocr_weights"].unique()) == {.2, 1., 2.}
        eos_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        assert weighted["ocr_weights"][weighted["ocr_targets"].eq(eos_id)].eq(2).all()
        assert weighted["ocr_weights"].sum() < weighted["ocr_targets"].numel()
        print("Grounding prompt, no thinking, EOS, left padding and shifted assistant-only targets verified.", flush=True)
        print("Native tokenizer offsets and position-weighted text/header/EOS supervision verified.", flush=True)


if __name__ == "__main__":
    main()
