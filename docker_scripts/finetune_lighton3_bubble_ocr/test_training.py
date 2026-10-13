"""Exercise shared numerical invariants and the native empty-prompt contract."""
import unittest
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import train
import torch
shared_spec = importlib.util.spec_from_file_location('surya_bubble_shared_tests', train.core.SCRIPT_DIR / 'test_training.py')
shared_tests = importlib.util.module_from_spec(shared_spec)
shared_spec.loader.exec_module(shared_tests)
SharedDataAndLossTests = shared_tests.DataAndLossTests


class PromptTests(unittest.TestCase):
    def test_transcription_keeps_visible_quotes_and_case(self):
        self.assertEqual(train.normalize_transcription('  "C\'est moi."  '), '"C\'est moi."')
        self.assertEqual(train.normalize_transcription("  'ARGH...' "), "'ARGH...'")
        self.assertEqual(train.core.postprocess_prediction('ARGH...'), 'ARGH...')

    def test_only_whitespace_is_normalized(self):
        self.assertEqual(train.normalize_transcription('Été\n\t!!  Deux...'), 'Été !! Deux...')

    def test_eos_weight_preserves_each_bubble_contribution(self):
        labels = torch.tensor([[-100, 4, 5, 9], [-100, -100, 3, 9]])
        weights = torch.tensor([1.2, 1.0])
        batch = train.weighted_packed_targets({"labels": labels}, weights, eos_token_id=9, eos_weight=2.0)
        torch.testing.assert_close(batch["ocr_token_weights"], torch.tensor([.3, .3, .6, 1/3, 2/3]))
        torch.testing.assert_close(batch["ocr_token_weights"].sum(), weights.sum())

    def test_image_only_and_plain_assistant_target(self):
        messages = train.messages_for_entry(Path('bubble.png'), 'BONJOUR !')
        self.assertEqual(messages[0]['content'], [{'type': 'image', 'image': 'bubble.png'}])
        self.assertEqual(messages[1]['content'][0]['text'], 'BONJOUR !')
        processor = Mock()
        train.apply_template(processor, messages, False)
        self.assertIs(processor.apply_chat_template.call_args.kwargs['enable_thinking'], False)
        self.assertEqual(train.core.USER_PROMPT, '')


if __name__ == '__main__':
    unittest.main()
