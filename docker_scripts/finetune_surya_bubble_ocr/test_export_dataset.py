import json
import tempfile
import unittest
from pathlib import Path

import export_dataset as export


class ReferenceSplitTests(unittest.TestCase):
    def write_reference(self, root, splits):
        for name, page_ids in splits.items():
            path = Path(root) / name / "metadata.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "".join(json.dumps({"page_id": page_id}) + "\n" for page_id in page_ids),
                encoding="utf-8",
            )

    def test_preserves_old_assignments_and_assigns_new_pages_stably(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_reference(root, {"train": ["1", "2"], "val": ["3"], "test": ["4"]})
            original = export.split_pages(["1", "2", "3", "4", "5", "6"], root)
            expanded = export.split_pages(["7", "6", "5", "4", "3", "2", "1"], root)
            self.assertIn("1", original["train"])
            self.assertIn("3", original["val"])
            self.assertIn("4", original["test"])
            for split, pages in original.items():
                self.assertTrue(set(pages).issubset(expanded[split]))
            self.assertEqual(sum(map(len, expanded.values())), 7)

    def test_rejects_leaking_and_incomplete_reference(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_reference(root, {"train": ["1"], "val": ["1"], "test": ["3"]})
            with self.assertRaisesRegex(ValueError, "appears in"):
                export.split_pages(["1", "2", "3"], root)
        with tempfile.TemporaryDirectory() as root:
            self.write_reference(root, {"train": ["1"], "val": ["2"]})
            with self.assertRaises(FileNotFoundError):
                export.split_pages(["1", "2", "3"], root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
