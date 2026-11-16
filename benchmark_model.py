"""Fresh-process isolated CPU measurements. Tokenization excluded from timed forward calls."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from engine import Engine, BACKENDS
from measure import MemorySampler, summary_ms, hardware
from workload import corpus, queries, performance_texts, workload_hash, ranking_metrics
from download_model import ROOT, MODEL, REVISION


def main(args):
    with MemorySampler() as memory:
        engine = Engine(args.backend, args.threads)
        texts = performance_texts()
        timings = {}
        for batch_size in (1, 8):
            batches = [engine.prepare([texts[(i * batch_size + j) % len(texts)] for j in range(batch_size)]) for i in range(args.iterations)]
            for i in range(5):
                engine.forward(batches[i % len(batches)])
            elapsed = []
            for batch in batches:
                start = time.perf_counter()
                output = engine.forward(batch)
                assert output.shape == (batch_size, 384) and np.isfinite(output).all()
                elapsed.append((time.perf_counter() - start) * 1000)
            timings[str(batch_size)] = {**summary_ms(elapsed), 'texts_per_second': batch_size * len(elapsed) / (sum(elapsed) / 1000),
                'min_padded_tokens': min(b['input_ids'].shape[1] for b in batches),
                'max_padded_tokens': max(b['input_ids'].shape[1] for b in batches)}
        docs = np.concatenate([engine.encode([r['text'] for r in corpus()[i:i+8]]) for i in range(0, len(corpus()), 8)])
        qs = np.concatenate([engine.encode([r['text'] for r in queries()[i:i+8]]) for i in range(0, len(queries()), 8)])
        quality = ranking_metrics(qs @ docs.T)
        vector_path = ROOT / 'reports' / (args.backend + '-vectors.npz')
        vector_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(vector_path, docs=docs, queries=qs)
    result = {'backend': args.backend, 'model': MODEL, 'revision': REVISION, 'hardware': hardware(),
        'threads': args.threads, 'workload_hash': workload_hash(), 'iterations': args.iterations,
        'warmup_per_batch_size': 5, 'timing': timings, 'quality': quality, 'peak_sampled_rss_bytes': memory.peak,
        'scope': 'Fresh worker process; warm forward only incl tensor conversion, excluding tokenization, HTTP and queueing. RSS includes Python/libraries/tokenizer/model; sampled every 10ms.'}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'backend': args.backend, 'timing': timings, 'quality': quality}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=BACKENDS, required=True)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--iterations', type=int, default=48)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if not 5 <= args.iterations <= 1000:
        p.error('Iterations must be 5–1000')
    main(args)
