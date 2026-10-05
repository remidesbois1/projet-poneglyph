"""Check real ORT execution after pruning/fusing, without a checkpoint or GPU."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T, helper as h, numpy_helper as nh
from export_prefill import export


class PrefillOptimizationTests(unittest.TestCase):
    def test_pruned_fused_graph_preserves_recurrent_output(self):
        rng = np.random.default_rng(17)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            initializers = [nh.from_array(rng.normal(size=16).astype(np.float32), "scale"),
                            nh.from_array(np.array(1e-6, np.float32), "epsilon")]
            nodes = [h.make_node("Mul", ["x", "x"], ["square"], name="square"),
                     h.make_node("ReduceMean", ["square"], ["mean"], axes=[-1], keepdims=1, name="mean"),
                     h.make_node("Add", ["mean", "epsilon"], ["shift"], name="shift"),
                     h.make_node("Sqrt", ["shift"], ["root"], name="root"),
                     h.make_node("Div", ["x", "root"], ["norm"], name="norm"),
                     h.make_node("Mul", ["norm", "scale"], ["present_state_0"], name="scaled"),
                     h.make_node("ArgMax", ["present_state_0"], ["next_token"], axis=-1, name="unused_lm_head")]
            graph = h.make_graph(nodes, "prefill-test", [h.make_tensor_value_info("x", T.FLOAT, [1, 3, 16])],
                [h.make_tensor_value_info("present_state_0", T.FLOAT, [1, 3, 16]), h.make_tensor_value_info("next_token", T.INT64, [1, 3, 1])], initializers)
            onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], ir_version=9), directory / "decoder.onnx")
            (directory / "browser_manifest.json").write_text(json.dumps({"files": {}}))
            feeds = {"x": rng.normal(size=(1, 3, 16)).astype(np.float32)}
            expected = ort.InferenceSession(str(directory / "decoder.onnx")).run(["present_state_0"], feeds)[0]
            for fused, filename in [(False, "prefill.onnx"), (True, "prefill-fast.onnx")]:
                export(directory, fused)
                optimized = onnx.load(directory / filename)
                self.assertEqual([x.name for x in optimized.graph.output], ["present_state_0"])
                self.assertNotIn("ArgMax", [x.op_type for x in optimized.graph.node])
                if fused:
                    self.assertEqual([x.op_type for x in optimized.graph.node], ["RMSNormalization"])
                actual = ort.InferenceSession(str(directory / filename)).run(None, feeds)[0]
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
