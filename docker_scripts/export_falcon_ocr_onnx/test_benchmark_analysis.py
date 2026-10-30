import unittest

from analyze_benchmark import compare, edit_distance, normalize


def row(text, reference, ms, limited=False):
    return {'text': text, 'reference': reference, 'timings': {'totalMs': ms}, 'tokenLimit': limited}


class BenchmarkAnalysisTests(unittest.TestCase):
    def test_compares_matching_ids_and_corpus_weighted_errors(self):
        old = {'a': row('é  oui', 'é oui', 20), 'b': row('non', 'nom', 40)}
        new = {'b': row('nom', 'nom', 10), 'a': row('e\u0301 oui', 'é oui', 5)}
        result = compare(old, new)
        self.assertEqual(result['identical_normalized'], 1)
        self.assertEqual(result['baseline']['cer'], 1 / 8)
        self.assertEqual(result['candidate']['cer'], 0)
        self.assertEqual(result['paired_median_speedup'], 4)
        self.assertEqual(result['max_output_character_edits'], 1)

    def test_rejects_different_samples(self):
        with self.assertRaises(ValueError):
            compare({'a': row('a', 'a', 1)}, {'b': row('a', 'a', 1)})

    def test_raw_comparison_preserves_untrimmed_text(self):
        old, new = row('salut', 'salut', 1), row('salut', 'salut', 1)
        old['rawText'], new['rawText'] = ' salut', 'salut'
        result = compare({'a': old}, {'a': new})
        self.assertEqual(result['identical_raw'], 0)
        self.assertEqual(result['identical_normalized'], 1)

    def test_unicode_and_sequence_distance(self):
        self.assertEqual(normalize('e\u0301\n oui'), 'é oui')
        self.assertEqual(edit_distance(['a', 'b'], ['a', 'c', 'd']), 2)


if __name__ == '__main__':
    unittest.main()
