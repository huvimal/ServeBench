"""Identical tokenizer, attention-mask mean pooling and L2 normalization on CPU."""
import os
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from download_model import ROOT, BASE, REVISION, verify, sha

BACKENDS = ('torch-fp32', 'onnx-fp32', 'onnx-int8')
MAX_TOKENS = 128


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = AutoModel.from_pretrained(str(BASE), local_files_only=True,
            trust_remote_code=False, use_safetensors=True, attn_implementation='eager').eval()

    def forward(self, input_ids, attention_mask, token_type_ids):
        states = self.base(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(states.dtype)
        pooled = (states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)


class Engine:
    def __init__(self, backend='onnx-int8', threads=2):
        if backend not in BACKENDS or not 1 <= threads <= 16:
            raise ValueError('Invalid backend or CPU thread count')
        verify()
        self.backend, self.threads = backend, threads
        self.tokenizer = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True, trust_remote_code=False)
        torch.set_num_threads(threads)
        if backend == 'torch-fp32':
            self.model = Encoder().eval()
        else:
            import onnxruntime as ort
            manifest = json.loads((ROOT / 'private/export.json').read_text(encoding='utf-8'))
            name = 'encoder-fp32.onnx' if backend == 'onnx-fp32' else 'encoder-int8.onnx'
            path = ROOT / 'private' / name
            if manifest['revision'] != REVISION or manifest['base_weights_sha256'] != sha(BASE / 'model.safetensors') or manifest['files'][name] != sha(path):
                raise ValueError('Export/base checksum mismatch')
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self.model = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])

    def prepare(self, texts):
        if not texts or len(texts) > 32 or any(not isinstance(t, str) or not t.strip() or len(t) > 2000 for t in texts):
            raise ValueError('Use 1–32 nonblank English texts, each at most 2000 characters')
        tokens = self.tokenizer(texts, padding=True, truncation=False, return_tensors='np')
        if tokens['input_ids'].shape[1] > MAX_TOKENS:
            raise ValueError('Text exceeds 128 tokens; no silent truncation')
        return {k: np.asarray(v, dtype=np.int64) for k, v in tokens.items()}

    def forward(self, tokens):
        if self.backend == 'torch-fp32':
            with torch.inference_mode():
                return self.model(**{k: torch.from_numpy(v) for k, v in tokens.items()}).numpy()
        return self.model.run(['embeddings'], tokens)[0]

    def encode(self, texts):
        return self.forward(self.prepare(texts))
