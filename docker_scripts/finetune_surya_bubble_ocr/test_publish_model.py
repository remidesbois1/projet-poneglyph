"""Regression checks for publishing only the verified, public release."""

import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import publish_model


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.source, self.onnx = root / "native", root / "onnx"
        self.source.mkdir()
        self.onnx.mkdir()
        for name in publish_model.NATIVE_FILES:
            (self.source / name).write_bytes(name.encode())
        self.digest = publish_model.sha256(self.source / "model.safetensors")
        benchmark = {"metrics": {"sample_count": 1, "token_limit_count": 0, "cer": 0,
                                  "wer": 0, "exact_match": 1},
                     "samples": [{"prediction": "PRIVATE PREDICTION"}],
                     "top_20_worst_errors": ["PRIVATE ERROR"]}
        (self.source / "benchmark_test.json").write_text(json.dumps(benchmark), encoding="utf-8")
        graph = b"example graph"
        (self.onnx / "vision.onnx").write_bytes(graph)
        (self.onnx / "fixtures").mkdir()
        (self.onnx / "fixtures" / "private.png").write_bytes(b"private")
        self.manifest = {"model_id": "owner/model", "source_sha256": self.digest,
                         "files": {"vision.onnx": {"bytes": len(graph), "sha256": hashlib.sha256(graph).hexdigest()}}}
        self.manifest_path = self.onnx / "browser_manifest.json"
        self.write_manifest()

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_mismatched_export_is_rejected_before_any_network_write(self):
        self.manifest["source_sha256"] = "0" * 64
        self.write_manifest()
        with patch.object(publish_model, "HfApi") as api:
            with self.assertRaisesRegex(ValueError, "does not match"):
                publish_model.publish(self.source, "owner/model", self.onnx, "parent")
            api.assert_not_called()

    def test_commit_excludes_fixtures_and_individual_predictions(self):
        remote_weight = types.SimpleNamespace(rfilename="model.safetensors", size=len(b"model.safetensors"),
                                              lfs=types.SimpleNamespace(sha256=self.digest))
        api = types.SimpleNamespace()
        api.model_info = lambda repo, **kwargs: (types.SimpleNamespace(sha="parent", siblings=[
            types.SimpleNamespace(rfilename="onnx/webgpu_metrics.json")]) if "revision" not in kwargs else
            types.SimpleNamespace(siblings=[remote_weight]))
        recorded = {}
        def commit(repo, **kwargs):
            recorded.update(kwargs)
            return types.SimpleNamespace(oid="revision", commit_url="https://example.org/revision")
        api.create_commit = commit
        with patch.object(publish_model, "HfApi", return_value=api), \
             patch.object(publish_model, "hf_hub_download", return_value=str(self.manifest_path)):
            publish_model.publish(self.source, "owner/model", self.onnx, "parent")
        self.assertEqual(recorded["parent_commit"], "parent")
        operations = recorded["operations"]
        self.assertFalse(any("fixtures" in op.path_in_repo for op in operations))
        benchmark = next(op for op in operations if op.path_in_repo == "benchmark_test.json")
        public = json.loads(benchmark.path_or_fileobj)
        self.assertNotIn("samples", public)
        self.assertNotIn("top_20_worst_errors", public)
        self.assertEqual(public["weights_sha256"], self.digest)


if __name__ == "__main__":
    unittest.main()
