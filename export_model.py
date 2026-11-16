"""Export actual pretrained encoder; dynamically quantize supported MatMul weights."""
import json
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from engine import Encoder, Engine
from download_model import ROOT, BASE, MODEL, REVISION, verify, sha


def main():
    verify()
    engine = Engine('torch-fp32')
    batch = engine.prepare(['A support request.', 'Where is my shipment?'])
    model = engine.model
    fp32, int8 = ROOT / 'private/encoder-fp32.onnx', ROOT / 'private/encoder-int8.onnx'
    torch.onnx.export(model, tuple(torch.from_numpy(batch[k]) for k in ('input_ids', 'attention_mask', 'token_type_ids')),
        str(fp32), input_names=['input_ids', 'attention_mask', 'token_type_ids'], output_names=['embeddings'],
        dynamic_axes={k: {0: 'batch', 1: 'sequence'} for k in batch} | {'embeddings': {0: 'batch'}},
        opset_version=17, dynamo=False, external_data=False)
    onnx.checker.check_model(str(fp32))
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, per_channel=True,
        op_types_to_quantize=['MatMul'])
    onnx.checker.check_model(str(int8))
    manifest = {'model': MODEL, 'revision': REVISION, 'base_weights_sha256': sha(BASE / 'model.safetensors'),
        'scope': 'CPU; dynamic INT8 MatMul weight quantization, not all operations are INT8. Mean pooling plus L2 normalization.',
        'opset': 17, 'files': {p.name: sha(p) for p in (fp32, int8)},
        'sizes_bytes': {p.name: p.stat().st_size for p in (fp32, int8)}}
    (ROOT / 'private/export.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
