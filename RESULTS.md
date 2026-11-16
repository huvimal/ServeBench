# Measured results — run-v2

Actual local measurements on 2026-11-16. Full frozen evidence: [benchmarks/run-v2.json](benchmarks/run-v2.json), including every HTTP latency/status sample, failures, versions, hardware, workload hash and exported-model checksums.

## Configuration

- Windows build 26200, Python 3.11.9, CPU identifier `AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD`; 4 physical cores / 8 logical CPUs, approximately 16 GB installed RAM.
- CPU only; two inference threads; one inference worker per server. GPU was not used.
- `sentence-transformers/paraphrase-MiniLM-L3-v2`, revision `4ca70771034acceecb2e72475f72050fcdde4ddc`; same tokenizer and normalized mean-pooling pipeline.
- PyTorch 2.10.0+cpu, Transformers 4.57.6, ONNX 1.23.2, ONNX Runtime 1.32.0. Exact environment: `requirements-lock-windows.txt`.
- 16 authored performance texts, 14–65 tokens individually; five warm-up calls per isolated batch size, then 48 timed calls. Timed forward includes framework input conversion but excludes tokenizer/HTTP/queueing.
- Six fresh API configurations: three backends × max batch 1/8, wait 0/5 ms, queue 16. Each receives three warm-ups and 80 measured requests at each concurrency level (1 and 8).

## Isolated warm forward

| Backend | Batch | p50 ms | p95 ms | Texts/sec |
|---|---:|---:|---:|---:|
| PyTorch FP32 | 1 | 10.27 | 14.37 | 95.08 |
| ONNX FP32 | 1 | 4.03 | 6.84 | 239.21 |
| ONNX INT8 | 1 | 2.96 | 4.81 | 344.06 |
| PyTorch FP32 | 8 | 39.52 | 62.83 | 198.37 |
| ONNX FP32 | 8 | 27.31 | 46.78 | 278.26 |
| ONNX INT8 | 8 | 23.96 | 44.62 | 377.94 |

These are sampled local timings, not guaranteed response times. Batch-8 latency is for eight texts together. Do not present it as one-request latency. No multi-run confidence intervals or controlled thermal/power conditions were established.

## Real loopback HTTP — concurrency 8

| Backend | Max batch / wait | Successful p50 ms | Successful p95 ms | Successful requests/sec | Errors |
|---|---|---:|---:|---:|---:|
| PyTorch FP32 | 1 / 0 ms | 99.59 | 119.09 | 76.28 | 0/80 |
| PyTorch FP32 | 8 / 5 ms | 93.02 | 217.48 | 68.46 | 0/80 |
| ONNX FP32 | 1 / 0 ms | 69.34 | 219.57 | 85.06 | 0/80 |
| ONNX FP32 | 8 / 5 ms | 61.21 | 305.64 | 78.37 | 0/80 |
| ONNX INT8 | 1 / 0 ms | 60.81 | 171.87 | 102.74 | 0/80 |
| ONNX INT8 | 8 / 5 ms | 59.82 | 79.46 | 69.26 | 0/80 |

All 960 normal measured requests across both concurrency levels returned valid 384-value finite embeddings with HTTP 200. Concurrency-1 results and full latency distributions remain in the report. These small samples can have long outliers beyond p95. p95 and total-run goodput describe different aspects of the run.

Batching did **not** increase successful throughput in these concurrency-8 comparisons. A max batch of 8 does not mean every model call contains 8 requests; actual batch-size distributions are preserved. The model, padding, FIFO batching window, local client scheduling and request distribution all matter. Do not replace this result with a generic batching-speedup claim.

## Quality and conversion drift

All documents and queries were encoded in batches of 8 for this quality check. Dynamic INT8 results can depend on batch composition; this does not establish quality under every HTTP batching pattern.

| Backend | Recall@1 | Recall@3 | MRR | Top-1 agreement with PyTorch |
|---|---:|---:|---:|---:|
| PyTorch FP32 | 18/24 | 23/24 | 0.8576 | Reference |
| ONNX FP32 | 18/24 | 23/24 | 0.8576 | 24/24 |
| ONNX INT8 | 21/24 | 23/24 | 0.9271 | 20/24 |

ONNX FP32 maximum absolute embedding difference from PyTorch was `1.19e-7` on 48 document/query vectors. INT8 mean cosine to PyTorch was 0.9859 (minimum 0.9719), with max absolute element difference 0.04722. Its higher Recall@1 on this tiny authored set is **not** evidence that quantization generally improves accuracy. Four top-1 results changed; rank changes must be evaluated on independent data before use.

The INT8 failures confuse receipt retrieval with refunds, shipment tracking with changing an address, and a timeout with rate limiting. FP32 has six top-1 failures, all preserved in the report. No model training, quality-based configuration selection or label changes followed result inspection.

## Storage and memory are different

Exported ONNX FP32: 69,022,162 bytes. INT8: 53,192,223 bytes, approximately **22.9% smaller**. Only supported MatMul weights were dynamically quantized; large embedding tables and other operations remain floating point. This is not a claim of a fully INT8 graph or 4× total compression.

Isolated sampled whole-process RSS peaks were approximately 351.8 MiB (PyTorch), 399.0 MiB (ONNX FP32), and 386.0 MiB (ONNX INT8). The smaller INT8 file did not yield a lower process RSS than PyTorch in this run.

HTTP-worker RSS peaks ranged approximately 351.6–404.4 MiB depending on backend/batching. The actual worker PID was checked against the owned process tree, and sampling began after readiness. Python, imported libraries and tokenizer are included; startup/load peaks, exact allocator peaks and GPU memory are not measured by this HTTP RSS figure.

## Overload and deadlines

A separate ONNX INT8/max-batch-8 probe sent 64 requests at concurrency 64 with a 1 ms server deadline. Outcomes: **18 HTTP 429 and 46 HTTP 504**, zero successful requests. Queue high-water mark was exactly 16, its configured capacity. Timed-out calls were not counted as successful throughput. No 500 or transport error occurred in that probe.

This intentionally impossible deadline checks rejection/expiry behavior; it is not a realistic service target, a maximum-capacity estimate or evidence of resilience to arbitrary attacks. A running native call finishes on the same worker after a caller timeout, with its late result discarded. Model work is not forcibly interrupted.

## Scope

One machine, one short local run, self-authored English inputs, no independent relevance adjudication, no production traffic, no external deployment or SLA. A meaningful next experiment would use independent domain queries, repeated runs and a controlled open-loop workload while retaining all failures.
