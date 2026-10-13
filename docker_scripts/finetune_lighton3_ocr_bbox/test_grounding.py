import numpy as np
import pytest

from grounding import aggregate, assign, format_grounding, page_metrics, parse_grounding


def test_native_grounding_roundtrip():
    items = [{"text": "Écoute, Luffy !", "bbox": [5, 10, 120, 160]}, {"text": "??", "bbox": [500, 510, 600, 620]}]
    text = format_grounding(items)
    parsed, invalid = parse_grounding(text)
    assert invalid == 0
    assert [(r["text"], r["bbox"]) for r in parsed] == [(r["text"], r["bbox"]) for r in items]
    metrics = aggregate([page_metrics(text, text)])
    assert metrics["cer"] == 0 and metrics["f1@0_5"] == 1 and metrics["combined_score"] == pytest.approx(1)


def test_missing_and_hallucinated_text_are_penalized():
    reference = format_grounding([{"text": "ABCD", "bbox": [0, 0, 100, 100]}])
    hallucination = format_grounding([{"text": "ABCD", "bbox": [0, 0, 100, 100]}, {"text": "FAUX", "bbox": [500, 500, 700, 700]}])
    assert page_metrics(reference, "")["cer"] == 1
    assert page_metrics(reference, hallucination)["cer"] == 1
    assert page_metrics(reference, hallucination)["precision@0_5"] == .5


def test_invalid_output_and_truncation():
    reference = format_grounding([{"text": "X", "bbox": [0, 0, 100, 100]}])
    metrics = page_metrics(reference, reference + "\n![text](500,500,1200,600)\nY", truncated=True)
    assert metrics["format_valid"] == 0
    assert metrics["precision@0_5"] == .5 and metrics["truncated"] == 1
    _, invalid = parse_grounding("![text](1,2,3)\nwrong")
    assert invalid >= 1


def test_matching_maximizes_detection_cardinality():
    # Highest-sum unconstrained assignment loses one eligible match here.
    matrix = np.array([[1.0, .51], [.51, .49]])
    assert set(assign(matrix, .5)) == {(0, 1), (1, 0)}


def test_reading_order_measured_separately():
    items = [{"text": "premier", "bbox": [10, 10, 100, 100]}, {"text": "second", "bbox": [200, 10, 300, 100]}]
    result = page_metrics(format_grounding(items), format_grounding(items[::-1]))
    assert result["cer"] == 0 and result["reading_order_cer"] > 0
    assert result["wer"] == 0 and result["reading_order_word_edits"] > 0
    assert result["page_text_exact"] == 0


def test_word_errors_include_missing_and_extra_regions():
    reference = format_grounding([{"text": "bonjour monde", "bbox": [0, 0, 100, 100]}])
    prediction = format_grounding([{"text": "bonjour ami", "bbox": [0, 0, 100, 100]},
                                  {"text": "faux ajout", "bbox": [500, 500, 700, 700]}])
    metrics = aggregate([page_metrics(reference, prediction)])
    assert metrics["wer"] == 1.5  # one substitution and two inserted words / two GT words
    assert metrics["exact_bubble_text_rate"] == 0
    assert page_metrics(reference, "")["wer"] == 1


def test_text_exact_ignores_coordinate_rounding_but_bbox_metric_requires_overlap():
    reference = format_grounding([{"text": "Salut !", "bbox": [0, 0, 100, 100]}])
    prediction = format_grounding([{"text": "Salut !", "bbox": [1, 1, 101, 101]}])
    metrics = aggregate([page_metrics(reference, prediction)])
    assert metrics["exact_match"] == 0
    assert metrics["page_text_exact"] == metrics["page_text_and_boxes_exact_at_0_5"] == 1
    distant = format_grounding([{"text": "Salut !", "bbox": [500, 500, 600, 600]}])
    result = page_metrics(reference, distant)
    assert result["page_text_exact"] == 1 and result["page_text_and_boxes_exact_at_0_5"] == 0


def test_wer_and_bubble_exact_use_corpus_denominators():
    short = format_grounding([{"text": "un", "bbox": [0, 0, 100, 100]}])
    long = format_grounding([{"text": "un deux trois", "bbox": [0, 0, 100, 100]},
                            {"text": "quatre", "bbox": [200, 200, 300, 300]}])
    metrics = aggregate([page_metrics(short, ""), page_metrics(long, long)])
    assert metrics["wer"] == pytest.approx(1 / 5)
    assert metrics["exact_bubble_text_rate"] == pytest.approx(2 / 3)
    assert metrics["page_text_exact"] == .5


def test_native_inline_blocks_and_image_description():
    prediction = "![image](0,0,1000,1000) Manga panel.\n\n![text](5,10,120,160) Écoute, Luffy !"
    reference = format_grounding([{"text": "Écoute, Luffy !", "bbox": [5, 10, 120, 160]}])
    result = page_metrics(reference, prediction)
    assert result["cer"] == 0 and result["f1@0_5"] == 1 and result["format_valid"] == 1


@pytest.mark.parametrize("box", [[0, 0, 0, 10], [-1, 0, 10, 10], [0, 0, 1001, 1002]])
def test_invalid_reference_rejected(box):
    with pytest.raises(ValueError):
        format_grounding([{"text": "x", "bbox": box}])
