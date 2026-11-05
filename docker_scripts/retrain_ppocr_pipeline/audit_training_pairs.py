"""Flag likely truncated training inputs for visual review; never mine test data."""
import argparse
import difflib
import json
import re
import unicodedata
from pathlib import Path


def comparable(text):
    return re.sub(r'[^\w]+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()


def audit(rows):
    suspects = []
    for row in rows:
        if '/train/' not in row['image']:
            raise ValueError('Only train-split predictions may be mined')
        ref, pred = comparable(row['reference']), comparable(row['raw'])
        steps = len(row['reference']) + sum(a == b for a, b in zip(row['reference'], row['reference'][1:]))
        if steps > row['timesteps']:
            suspects.append({**row, 'reason': 'impossible_ctc_alignment', 'required_steps': steps})
            continue
        if len(ref) < 25 or len(pred) < 12 or len(pred) > .8 * len(ref):
            continue
        operations = difflib.SequenceMatcher(None, ref, pred, autojunk=False).get_opcodes()
        matched = sum(j-i for tag,i,j,_,_ in operations if tag == 'equal')
        missing = [ref[i:j] for tag,i,j,_,_ in operations if tag == 'delete' and len(ref[i:j].split()) >= 2 and sum(c.isalpha() for c in ref[i:j]) >= 8]
        if matched >= .9 * len(pred) and missing:
            suspects.append({**row, 'reason': 'possible_missing_visual_line', 'missing_segments': missing,
                             'deleted_characters': sum(len(x) for x in missing)})
    return suspects


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(s) for s in args.predictions.read_text(encoding='utf-8').splitlines() if s]
    suspects = audit(rows)
    args.output.write_text(json.dumps(suspects, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'train_rows': len(rows), 'suspected_pairs_for_visual_review': len(suspects)}))
