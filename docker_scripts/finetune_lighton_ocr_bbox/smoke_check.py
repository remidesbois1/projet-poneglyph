import inspect
import os

import torch
import transformers
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

import train_lighton_bbox as bbox


MODEL_ID = os.getenv("LIGHTON_BBOX_MODEL_ID", "lightonai/LightOnOCR-2-1B-bbox-base")
EDGE = int(os.getenv("LIGHTON_BBOX_IMAGE_LONGEST_EDGE", "1500"))


def main():
    assert tuple(int(part) for part in transformers.__version__.split(".")[:2]) >= (5, 14), transformers.__version__
    assert EDGE == 1500, "LightOn bbox training must keep the 1500 px page profile"
    assert "[x1,y1,x2,y2]" in bbox.OUTPUT_CONTRACT
    assert "N'ajoute aucun JSON" in bbox.OUTPUT_CONTRACT
    assert bbox.MODEL_ID == MODEL_ID
    assert "logits_to_keep" in inspect.signature(LightOnOcrForConditionalGeneration.forward).parameters

    processor = LightOnOcrProcessor.from_pretrained(MODEL_ID)
    processor.image_processor.default_to_square = False
    processor.image_processor.size = {"longest_edge": EDGE}
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    print(
        f"Runtime OK: transformers={transformers.__version__}, model={MODEL_ID}, "
        f"longest_edge={EDGE}, selective logits supported.",
        flush=True,
    )

    if not bbox.env_bool("LIGHTON_BBOX_SMOKE_LOAD_MODEL", False):
        return
    if not torch.cuda.is_available():
        raise RuntimeError("LIGHTON_BBOX_SMOKE_LOAD_MODEL=1 requires CUDA")

    model = bbox.load_lighton_model(for_training=True)
    model, _ = bbox.configure_trainable_model(model)
    bbox.configure_gradient_checkpointing(model, False)
    print("BF16 LightOn model + rsLoRA initialization OK.", flush=True)


if __name__ == "__main__":
    main()
