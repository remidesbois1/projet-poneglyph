"""Compare private Falcon benchmark runs without exporting image or text data."""
import argparse
import json
import math
import statistics
import unicodedata
from pathlib import Path


def normalize(value):
    return ' '.join(unicodedata.normalize('NFC', value).split())


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def load(path):
    result = json.loads(path.read_text(encoding='utf-8'))
    rows = result['rows']
    ids = [str(row['id']) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError('Benchmark must contain distinct, nonempty IDs.')
    return {str(row['id']): row for row in rows}


def summarize(rows):
    total_errors = total_chars = word_errors = total_words = exact = 0
    for row in rows.values():
        expected, actual = normalize(row['reference']), normalize(row['text'])
        total_errors += edit_distance(expected, actual); total_chars += len(expected)
        expected_words, actual_words = expected.split(), actual.split()
        word_errors += edit_distance(expected_words, actual_words); total_words += len(expected_words)
        exact += expected == actual
    elapsed = sorted(row['timings']['totalMs'] for row in rows.values())
    return {'count': len(rows), 'median_ms': statistics.median(elapsed), 'mean_ms': statistics.mean(elapsed),
            'p95_ms': elapsed[math.ceil(len(elapsed) * .95) - 1], 'cer': total_errors / max(1, total_chars),
            'wer': word_errors / max(1, total_words), 'exact': exact,
            'truncated': sum(bool(row['tokenLimit']) for row in rows.values())}


def compare(baseline, candidate):
    if set(baseline) != set(candidate):
        raise ValueError('Benchmark runs must contain exactly the same IDs.')
    if any(normalize(baseline[key]['reference']) != normalize(candidate[key]['reference']) for key in baseline):
        raise ValueError('Benchmark references changed between runs.')
    edits = [edit_distance(normalize(baseline[key]['text']), normalize(candidate[key]['text'])) for key in baseline]
    ratios = [baseline[key]['timings']['totalMs'] / candidate[key]['timings']['totalMs']
              for key in baseline if candidate[key]['timings']['totalMs'] > 0]
    return {'baseline': summarize(baseline), 'candidate': summarize(candidate),
            'identical_raw': sum(baseline[key].get('rawText', baseline[key]['text']) ==
                                 candidate[key].get('rawText', candidate[key]['text']) for key in baseline),
            'identical_normalized': sum(value == 0 for value in edits), 'different': sum(value > 0 for value in edits),
            'termination_disagreements': sum(bool(baseline[key]['tokenLimit']) != bool(candidate[key]['tokenLimit']) for key in baseline),
            'max_output_character_edits': max(edits), 'paired_median_speedup': statistics.median(ratios)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = json.dumps(compare(load(args.baseline), load(args.candidate)), indent=2, ensure_ascii=False) + '\n'
    if args.output:
        args.output.write_text(report, encoding='utf-8')
    print(report, end='')


if __name__ == '__main__':
    main()
