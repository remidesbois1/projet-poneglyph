"""Pure regression tests for persisted Surya bbox pipeline state."""

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SOURCE = Path(__file__).with_name("run_pipeline.py")


def symbols(*names):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if len(selected) != len(names):
        raise AssertionError("A production pipeline symbol under test is missing.")
    scope = {"Path": Path, "json": json}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


class DatasetReadinessTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.api = symbols("dataset_readiness", "dataset_is_ready")

    def test_zero_byte_metadata_is_not_ready(self):
        for split in ("train", "val", "test"):
            split_dir = self.root / split
            split_dir.mkdir(parents=True)
            (split_dir / "metadata.jsonl").touch()

        ready, reason = self.api.dataset_readiness(self.root)
        self.assertFalse(ready)
        self.assertIn("empty", reason)
        self.assertFalse(self.api.dataset_is_ready(self.root))

    def test_valid_first_sample_and_image_are_ready(self):
        for split in ("train", "val", "test"):
            split_dir = self.root / split
            image_dir = split_dir / "images"
            image_dir.mkdir(parents=True)
            (image_dir / "page_1.jpg").write_bytes(b"image")
            sample = {"page_id": "1", "image_file": "images/page_1.jpg"}
            (split_dir / "metadata.jsonl").write_text(
                json.dumps(sample) + "\n", encoding="utf-8"
            )

        ready, reason = self.api.dataset_readiness(self.root)
        self.assertTrue(ready, reason)
        self.assertTrue(self.api.dataset_is_ready(self.root))


if __name__ == "__main__":
    unittest.main()
