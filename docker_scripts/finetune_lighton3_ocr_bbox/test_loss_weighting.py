import pytest

from loss_weighting import token_weights


def test_bbox_positions_not_digit_vocabulary_define_low_weight():
    reference = "![text](1,2,30,40)\n2026 : Hé !"
    start = 50
    offsets = [(start + i, start + i + 1) for i in range(len(reference))] + [(0, 0)]
    ids = list(range(len(reference))) + [999]
    weights = token_weights(reference, offsets, start, ids, 999, .2, 2.)
    header_end = reference.index(")") + 1
    assert weights[:header_end] == [.2] * header_end
    assert weights[header_end:-1] == [1.] * (len(reference) - header_end)
    assert weights[-1] == 2.


def test_token_overlapping_header_boundary_is_counted_once():
    weights = token_weights("![text](1,2,3,4)\nSalut", [(0, 18), (18, 22)], 0, [10, 20], 99, .2, 2.)
    assert weights == [.2, 1.]


def test_invalid_weights_rejected():
    with pytest.raises(ValueError):
        token_weights("X", [(0, 1)], 0, [1], 2, 0, 2)
