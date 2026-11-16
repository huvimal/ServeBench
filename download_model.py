"""Download only pinned public files; verify weights against upstream LFS SHA-256."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = 'sentence-transformers/paraphrase-MiniLM-L3-v2'
REVISION = '4ca70771034acceecb2e72475f72050fcdde4ddc'
BASE = ROOT / 'private/base'
WEIGHT_SHA = 'cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433'
FILES = ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt', 'README.md')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify():
    manifest = json.loads((BASE / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['model'] != MODEL or manifest['revision'] != REVISION or set(manifest['files']) != set(FILES):
        raise ValueError('Unexpected pinned model manifest')
    for name, info in manifest['files'].items():
        if sha(BASE / name) != info['sha256']:
            raise ValueError(f'Model checksum mismatch: {name}')
    if manifest['files']['model.safetensors']['sha256'] != WEIGHT_SHA:
        raise ValueError('Weights do not match the upstream pinned model')
    return manifest


def download():
    BASE.mkdir(parents=True, exist_ok=True)
    manifest = {'model': MODEL, 'revision': REVISION, 'files': {}}
    for name in FILES:
        path = BASE / name
        url = f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}'
        if path.exists() and (name != 'model.safetensors' or sha(path) == WEIGHT_SHA):
            pass
        else:
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(url, timeout=120) as response, path.with_suffix(path.suffix + '.part').open('wb') as f:
                        for block in iter(lambda: response.read(1024 * 1024), b''):
                            f.write(block)
                    path.with_suffix(path.suffix + '.part').replace(path)
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2)
        if name == 'model.safetensors' and sha(path) != WEIGHT_SHA:
            raise ValueError('Downloaded weights checksum mismatch')
        manifest['files'][name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
        print('Verified', name, flush=True)
    (BASE / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    verify()


if __name__ == '__main__':
    download()
