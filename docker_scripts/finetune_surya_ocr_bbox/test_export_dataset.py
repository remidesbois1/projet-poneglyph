"""Pure regression tests for the Surya bbox dataset exporter."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote, unquote, urlsplit


SOURCE = Path(__file__).with_name("export_dataset.py")


def symbols(*names):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if len(selected) != len(names):
        raise AssertionError("A production exporter symbol under test is missing.")
    scope = {"quote": quote, "unquote": unquote, "urlsplit": urlsplit}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


class R2ReferenceTests(unittest.TestCase):
    def test_private_r2_reference_contract(self):
        api = symbols("parse_r2_reference")
        self.assertEqual(
            api.parse_r2_reference(
                "r2://poneglyph-pages-private/tome-27/page%201.avif",
                "poneglyph-pages-private",
            ),
            ("poneglyph-pages-private", "tome-27/page 1.avif"),
        )

    def test_rejects_noncanonical_or_wrong_bucket_references(self):
        api = symbols("parse_r2_reference")
        invalid = (
            "r2://other-bucket/page.avif",
            "r2://poneglyph-pages-private/../page.avif",
            "r2://poneglyph-pages-private/foo%2Fbar.avif",
            "r2://poneglyph-pages-private/page.avif?download=1",
            " r2://poneglyph-pages-private/page.avif",
        )
        for reference in invalid:
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                api.parse_r2_reference(reference, "poneglyph-pages-private")


if __name__ == "__main__":
    unittest.main()
