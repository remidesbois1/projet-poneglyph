"""Build a verified, small GPU weight index for an existing Falcon ONNX export."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(model_dir, output):
    manifest = json.loads((model_dir / 'browser_manifest.json').read_text(encoding='utf-8'))
    graph_path, data_path = model_dir / 'decoder.onnx', model_dir / 'decoder.onnx.data'
    for path in (graph_path, data_path):
        info = manifest['files'][path.name]
        if path.stat().st_size != info['bytes'] or digest(path) != info['sha256']:
            raise ValueError(f'Falcon ONNX checksum mismatch: {path.name}')
    config = manifest['config']
    if (config['dim'], config['n_layers'], config['n_heads'], config['n_kv_heads'], config['ffn_dim']) != (768, 22, 16, 8, 2304):
        raise ValueError('Unsupported Falcon architecture')
    initializers = {x.name: x for x in onnx.load(str(graph_path), load_external_data=False).graph.initializer}
    expected = {'tok_embeddings.weight': ((65536, 768), TensorProto.FLOAT16, False),
                'output.weight': ((768, 65536), TensorProto.FLOAT16, True),
                'norm.weight': ((768,), TensorProto.FLOAT16, False)}
    for layer in range(config['n_layers']):
        prefix = f'layers.{layer}.'
        expected.update({prefix + 'attention.wqkv.weight': ((768, 2048), TensorProto.FLOAT16, True),
                         prefix + 'attention.wo.weight': ((1024, 768), TensorProto.FLOAT16, True),
                         prefix + 'attention.sinks': ((16,), TensorProto.FLOAT16, False),
                         prefix + 'feed_forward.w13.weight': ((768, 4608), TensorProto.FLOAT16, True),
                         prefix + 'feed_forward.w2.weight': ((2304, 768), TensorProto.FLOAT, True)})
    weights = {}
    for name, (shape, dtype, transpose) in expected.items():
        tensor = initializers[name]
        if tuple(tensor.dims) != shape or tensor.data_type != dtype:
            raise ValueError(f'Unexpected tensor shape or dtype: {name}')
        entry = {'shape': shape, 'dtype': 'float16' if dtype == TensorProto.FLOAT16 else 'float32', 'transpose': transpose}
        if tensor.data_location == TensorProto.EXTERNAL:
            location = {v.key: v.value for v in tensor.external_data}
            offset, size = int(location['offset']), int(location['length'])
            if location['location'] != data_path.name or offset < 0 or offset + size > data_path.stat().st_size:
                raise ValueError(f'Invalid weight data range: {name}')
            if size != int(np.prod(shape)) * (2 if dtype == TensorProto.FLOAT16 else 4):
                raise ValueError(f'Invalid weight byte length: {name}')
            entry.update(offset=offset, bytes=size)
        else:
            if dtype != TensorProto.FLOAT16:
                raise ValueError(f'Unexpected embedded tensor: {name}')
            entry['bits'] = numpy_helper.to_array(tensor).view(np.uint16).tolist()
        weights[name] = entry
    result = {'schema': 1, 'source_sha256': manifest['source_sha256'],
              'graph_sha256': manifest['files']['decoder.onnx']['sha256'],
              'data_sha256': manifest['files']['decoder.onnx.data']['sha256'], 'weights': weights}
    output.write_text(json.dumps(result, separators=(',', ':')) + '\n', encoding='utf-8')
    print(f'Wrote {len(weights)} verified tensors to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    build(args.model, args.output)
