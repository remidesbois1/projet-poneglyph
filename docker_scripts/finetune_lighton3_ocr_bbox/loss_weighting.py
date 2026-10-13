"""Position-aware OCR loss weights: text first, native grounding still supervised."""

from grounding import HEADER


def token_weights(reference, offsets, reference_start, token_ids, eos_id, bbox_weight=1.0, eos_weight=1.0):
    if len(offsets) != len(token_ids) or not 0 < bbox_weight <= 1 or eos_weight <= 0:
        raise ValueError("Invalid token offsets or OCR loss weights")
    headers = [(reference_start + m.start(), reference_start + m.end()) for m in HEADER.finditer(reference)]
    return [eos_weight if token == eos_id else bbox_weight if any(end > left and start < right for left, right in headers) else 1.0
            for (start, end), token in zip(offsets, token_ids)]
