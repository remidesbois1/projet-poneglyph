"""Checks the training boundary: exact production inputs and valid CTC targets."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'docker_scripts/finetune_paddleocr_line_rec'))
from train_ppocrv6_rec import RecDataset, checked_ctc_loss, make_collate


class TrainingBoundaryTests(unittest.TestCase):
    def test_png_reconstructs_production_tensor(self):
        fixture = Path(__file__).parent / 'outputs/parity'
        image = Image.open(fixture / 'input.png').convert('RGB')
        item = dict(image=image, target=torch.tensor([1, 1]), text='AA', loss_weight=1., is_short=True, is_single_line=True, is_hard=False)
        batch = make_collate(None, prepared_inputs=True)([item])
        expected = np.fromfile(fixture / 'tensor.f32', dtype='<f4').reshape(1, 3, 48, 960)
        # JS evaluates in float64 before writing float32. Torch's arithmetic is
        # float32; their difference must stay within one float32 rounding unit.
        np.testing.assert_allclose(batch['pixel_values'].numpy(), expected, atol=6e-8, rtol=0)
        self.assertEqual(batch['ctc_min_steps'].tolist(), [3])

    def test_repeated_characters_need_blank_timesteps(self):
        ds = RecDataset([(Path('x'), 'AA', 'x')], {'A': 1}, 2, 12, 1.)
        self.assertEqual(len(ds), 0)
        self.assertEqual(ds.excluded_rows[0]['required_steps'], 3)
        probabilities = torch.tensor([[[.5, .5]], [[.5, .5]]], requires_grad=True)
        with self.assertRaisesRegex(ValueError, 'timesteps'):
            checked_ctc_loss(probabilities.log(), torch.tensor([1, 1]), torch.tensor([2]), torch.tensor([2]), {'ctc_min_steps': torch.tensor([3]), 'texts': ['AA']})

    def test_unknown_characters_are_not_silently_deleted(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            RecDataset([(Path('x'), 'A?', 'x')], {'A': 1}, 10, 12, 1.)

    def test_valid_ctc_has_finite_nonzero_gradients(self):
        logits = torch.tensor([[[0., 1.]], [[1., 0.]], [[0., 1.]]], requires_grad=True)
        losses = checked_ctc_loss(logits.log_softmax(-1), torch.tensor([1, 1]), torch.tensor([3]), torch.tensor([2]), {'ctc_min_steps': torch.tensor([3]), 'texts': ['AA']})
        losses.mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
