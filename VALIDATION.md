# Verification — 2026-11-16

- **Dependency check:** `python -m pip check` passed; exact installed packages recorded in `requirements-lock-windows.txt`.
- **15 tests passed:** batch/result ordering, partial-batch flushing, bounded queue rejection, timeout without additional inference workers, cancellation, recovery from an inference failure, per-item failure isolation, API validation and metric/ranking arithmetic. API tests explicitly use a fake engine and are not model-quality measurements.
- **Actual weights:** pinned pretrained safetensors downloaded and verified against upstream SHA-256. Local source manifest checks all required model/tokenizer files.
- **Actual export:** PyTorch encoder exported at ONNX opset 17 using the explicitly selected legacy exporter (`dynamo=False`); ONNX checker passed. Dynamic per-channel QInt8 MatMul quantization completed, and the quantized graph also passed the checker. The exporter/quantizer emitted advisory deprecation/preprocessing warnings; the measured artifacts were still loaded and executed successfully. No claim is made that this is the newest export pipeline or globally optimized graph.
- **Actual model execution:** all three backends ran in fresh processes on the same CPU workload. Normalized embeddings and output dimensions were checked. FP32/INT8 conversion parity and retrieval quality were measured, including failures.
- **Actual HTTP:** six Uvicorn server configurations ran over loopback sockets. 960 normal measured requests succeeded; an additional 64-request overload/deadline probe exercised 429/504. Blank and over-token-limit inputs returned 422 on every configuration. Each backend also returned an actual 384-dimensional finite unit-length vector through HTTP.
- **Actual memory target:** API worker PID returned by health was checked against the launched process tree before sampling its RSS. Every measured HTTP worker exceeded 100 MB sampled RSS, avoiding accidental launcher-only figures. All six worker PIDs were confirmed exited after the benchmark.
- **Actual UI engine:** `verify_local.py` ran the ONNX INT8 model in Streamlit's app runner. The query about stopping monthly renewals returned `cancel-subscription` first, with three ranked documents and no application exception. This is not a browser screenshot/visual-layout check.

Frozen full report: [benchmarks/run-v2.json](benchmarks/run-v2.json). Read [RESULTS.md](RESULTS.md) for the measured tradeoffs and limitations. Run-v2 is the verified report after correcting Windows worker-PID memory attribution; an earlier diagnostic report remains local and is marked invalid for memory claims. The quality workload, labels and model exports were not tuned after inspecting results.

Docker build/execution, hosted GitHub Actions, live-browser visual layout, public deployment, sustained load, GPU inference and real-customer quality were not verified. The provided Dockerfile/CI workflow are scaffolds. Docker requires separately prepared CPU artifacts mounted into `/app/private`; these are excluded from the image build context.

## Local recheck

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -X utf8 verify_local.py
.\.venv\Scripts\python.exe -X utf8 run_benchmarks.py --output reports/my-new-run.json
```

Use a new final report filename to retain earlier results. CPU timings naturally vary. Source/model artifacts remain local under `private/`; runtime logs, worker intermediates and UI smoke evidence are under ignored `reports/`. The public frozen report contains synthetic inputs, model metrics and hardware/configuration only.
