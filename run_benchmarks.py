"""Sequential fresh-process model and real-loopback HTTP benchmarks; never a mock provider."""
import argparse
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import httpx
import numpy as np
import psutil
from download_model import ROOT, MODEL, REVISION, verify, sha
from engine import BACKENDS
from load_test import run_load
from measure import MemorySampler, hardware
from workload import corpus, performance_texts, workload_hash


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def wait_ready(process, url):
    deadline = time.perf_counter() + 90
    with httpx.Client(timeout=1, trust_env=False) as client:
        while time.perf_counter() < deadline:
            if process.poll() is not None:
                raise RuntimeError('API exited; inspect reports/server-*.log')
            try:
                response = client.get(url + '/health')
                if response.status_code == 200 and response.json()['ready']:
                    return response.json()
            except httpx.HTTPError:
                pass
            time.sleep(.2)
    raise RuntimeError('API readiness deadline exceeded')


def http_benchmark(backend, batch, count):
    port = free_port()
    url = f'http://127.0.0.1:{port}'
    env = {**os.environ, 'SERVEBENCH_BACKEND': backend, 'SERVEBENCH_BATCH': str(batch),
        'SERVEBENCH_WAIT_MS': '5' if batch > 1 else '0', 'SERVEBENCH_THREADS': '2', 'SERVEBENCH_QUEUE': '16'}
    with (ROOT / f'reports/server-{backend}-batch{batch}.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'api:app', '--host', '127.0.0.1',
            '--port', str(port), '--workers', '1', '--no-access-log'], cwd=ROOT, env=env, stdout=log, stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            ready = wait_ready(process, url)
            server_pid = ready['pid']
            owned_pids = [process.pid] + [p.pid for p in psutil.Process(process.pid).children(recursive=True)]
            if server_pid not in owned_pids or ready['backend'] != backend:
                raise RuntimeError('Ready server does not belong to this benchmark process tree')
            with MemorySampler(server_pid) as memory:
                with httpx.Client(base_url=url, timeout=15, trust_env=False) as client:
                    for text in performance_texts()[:3]:
                        client.post('/embed', json={'text': text}).raise_for_status()
                loads = [asyncio.run(run_load(url, count, c, 2000)) for c in (1, 8)]
                checks = {}
                with httpx.Client(base_url=url, timeout=15, trust_env=False) as client:
                    checks['blank_status'] = client.post('/embed', json={'text': ' '}).status_code
                    checks['over_token_limit_status'] = client.post('/embed', json={'text': 'word ' * 200}).status_code
                    assert checks == {'blank_status': 422, 'over_token_limit_status': 422}
                    # Actual network/model integration, not just an in-process test client.
                    actual = client.post('/embed', json={'text': corpus()[0]['text']})
                    actual.raise_for_status()
                    vector = np.asarray(actual.json()['embedding'])
                    assert vector.shape == (384,) and np.isfinite(vector).all() and abs(np.linalg.norm(vector) - 1) < .001
                    checks['actual_vector_dimension'] = len(vector)
                    checks['actual_vector_norm'] = float(np.linalg.norm(vector))
                stress = asyncio.run(run_load(url, 64, 64, 1)) if backend == 'onnx-int8' and batch == 8 else None
                if stress:
                    assert stress['statuses'].get('429', 0) > 0 and stress['statuses'].get('504', 0) > 0
                # Wait for completed/timed-out work to drain before recording cumulative metrics.
                time.sleep(.2)
                with httpx.Client(base_url=url, timeout=15, trust_env=False) as client:
                    metrics = client.get('/metrics').json()
                assert metrics['max_queue_seen'] <= metrics['queue_capacity']
                assert metrics['max_batch_seen'] <= batch
            return {'backend': backend, 'max_batch': batch, 'batch_wait_ms': 5 if batch > 1 else 0,
                'queue_capacity': 16, 'threads': 2, 'warmup_requests': 3, 'loads': loads,
                'overload_and_deadline_probe': stress, 'integration_checks': checks,
                'cumulative_server_metrics': metrics, 'server_peak_sampled_rss_bytes': memory.peak,
                'memory_scope': 'Actual API worker PID returned by health and validated against owned process tree; sampled after readiness through warmup/load/probes, excluding startup/load peak.'}
        finally:
            try:
                children = psutil.Process(process.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                children = []
            for child in reversed(children):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(children, timeout=10)
            for child in alive:
                child.kill()
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError('Output exists. Choose a new report filename to preserve prior measurements.')
    (ROOT / 'reports').mkdir(exist_ok=True)
    verify()
    isolated = []
    for backend in BACKENDS:
        target = ROOT / f'reports/{backend}-isolated.json'
        print('Measuring actual model:', backend, flush=True)
        result = subprocess.run([sys.executable, '-X', 'utf8', 'benchmark_model.py', '--backend', backend,
            '--iterations', str(args.iterations), '--output', str(target)], cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        (ROOT / f'reports/{backend}-isolated.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'{backend} benchmark failed; inspect reports/{backend}-isolated.log')
        isolated.append(json.loads(target.read_text(encoding='utf-8')))
    reference = np.load(ROOT / 'reports/torch-fp32-vectors.npz')
    parity = {}
    for backend in BACKENDS[1:]:
        candidate = np.load(ROOT / f'reports/{backend}-vectors.npz')
        left = np.concatenate([reference['docs'], reference['queries']])
        right = np.concatenate([candidate['docs'], candidate['queries']])
        cosine = np.sum(left * right, axis=1) / (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1))
        parity[backend] = {'vectors': len(left), 'mean_cosine_to_torch': float(cosine.mean()),
            'min_cosine_to_torch': float(cosine.min()), 'max_absolute_difference': float(np.abs(left - right).max()),
            'top1_agreement_to_torch': float(np.mean((candidate['queries'] @ candidate['docs'].T).argmax(1) ==
                (reference['queries'] @ reference['docs'].T).argmax(1)))}
    http = []
    for backend in BACKENDS:
        for batch in (1, 8):
            print('Measuring real HTTP:', backend, 'max batch', batch, flush=True)
            http.append(http_benchmark(backend, batch, args.requests))
    report = {'model': MODEL, 'revision': REVISION, 'workload_hash': workload_hash(), 'hardware': hardware(),
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'onnx', 'onnxruntime', 'fastapi', 'uvicorn', 'numpy', 'httpx']},
        'export': json.loads((ROOT / 'private/export.json').read_text(encoding='utf-8')),
        'isolated': isolated, 'parity': parity, 'http': http,
        'scope': 'Single local CPU machine, synthetic English retrieval set, warm model, closed-loop HTTP; no cloud deployment, GPU performance, production traffic or universal speedup claim.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved actual measurements:', output, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='reports/run-v1.json')
    p.add_argument('--iterations', type=int, default=48)
    p.add_argument('--requests', type=int, default=80)
    args = p.parse_args()
    if not 5 <= args.iterations <= 1000 or not 10 <= args.requests <= 10000:
        p.error('Invalid bounded workload')
    main(args)
