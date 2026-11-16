"""Closed-loop HTTP load generator. No silent retries; reports all statuses and goodput."""
import argparse
import asyncio
from collections import Counter
import json
import math
import time
from pathlib import Path
import httpx
from measure import summary_ms
from workload import performance_texts, workload_hash


async def run_load(url, count=80, concurrency=8, timeout_ms=2000):
    if not 1 <= count <= 10000 or not 1 <= concurrency <= 256:
        raise ValueError('Invalid load bounds')
    texts = performance_texts()
    results, index = [], 0
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=url, timeout=15, limits=limits, trust_env=False) as client:
        start = time.perf_counter()
        async def worker():
            nonlocal index
            while index < count:
                i = index
                index += 1
                t0 = time.perf_counter()
                try:
                    response = await client.post('/embed', json={'text': texts[i % len(texts)], 'timeout_ms': timeout_ms})
                    status = str(response.status_code)
                    try:
                        payload = response.json()
                    except ValueError:
                        payload = {}
                        status = 'invalid_response'
                    if status == '200':
                        embedding = payload.get('embedding')
                        if not isinstance(embedding, list) or len(embedding) != 384 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in embedding):
                            status = 'invalid_response'
                    batch = payload.get('batch_size') if status == '200' else None
                except httpx.HTTPError:
                    status, batch = 'transport_error', None
                results.append({'index': i, 'status': status, 'latency_ms': (time.perf_counter() - t0) * 1000, 'batch_size': batch})
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        elapsed = time.perf_counter() - start
    statuses = dict(Counter(r['status'] for r in results))
    successful = [r for r in results if r['status'] == '200']
    return {'requests': count, 'concurrency': concurrency, 'request_timeout_ms': timeout_ms,
        'seconds': elapsed, 'statuses': statuses, 'error_rate': 1 - len(successful) / count,
        'attempts_per_second': count / elapsed, 'successful_requests_per_second': len(successful) / elapsed,
        'all_attempt_latency': summary_ms([r['latency_ms'] for r in results]),
        'successful_latency': summary_ms([r['latency_ms'] for r in successful]),
        'observed_batch_sizes': dict(Counter(r['batch_size'] for r in successful)),
        'workload_hash': workload_hash(), 'samples': sorted(results, key=lambda r: r['index']),
        'scope': 'Closed-loop concurrent workers, keep-alive HTTP, no retries. Client wall time includes transport/queue/tokenization/model/serialization. Not an open-loop capacity or distributed-scale test.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8520')
    p.add_argument('--requests', type=int, default=80)
    p.add_argument('--concurrency', type=int, default=8)
    p.add_argument('--timeout-ms', type=int, default=2000)
    p.add_argument('--output', default='reports/http-custom.json')
    a = p.parse_args()
    result = asyncio.run(run_load(a.url, a.requests, a.concurrency, a.timeout_ms))
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'samples'}, indent=2))
