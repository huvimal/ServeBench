import json
import time
import numpy as np
import streamlit as st
from pathlib import Path
from workload import corpus

ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_search(backend):
    from engine import Engine
    model = Engine(backend)
    docs = corpus()
    vectors = np.concatenate([model.encode([d['text'] for d in docs[i:i+8]]) for i in range(0, len(docs), 8)])
    import threading
    return model, vectors, threading.Lock()


def main():
    st.set_page_config(page_title='ServeBench', layout='wide')
    st.title('ServeBench')
    st.caption('CPU model serving • PyTorch / ONNX / INT8 • measured quality and performance')
    st.info('Local English embedding demo. Measurements use a small synthetic workload on one CPU machine. No production-scale or universal speedup claim.')
    path = ROOT / 'reports/run-v2.json'
    if not path.exists():
        path = ROOT / 'benchmarks/run-v2.json'
    if path.exists():
        report = json.loads(path.read_text(encoding='utf-8'))
        st.caption(f'Measurement source: {path.parent.name}/{path.name}; frozen local CPU experiment.')
        manifest_path = ROOT / 'private/export.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8'))['files'] != report['export']['files']:
            st.warning('Measured export hashes differ from current local artifacts. Re-evaluate before comparing performance.')
        st.subheader('Isolated forward measurements')
        rows = []
        for variant in report['isolated']:
            for batch, timing in variant['timing'].items():
                rows.append({'backend': variant['backend'], 'batch': int(batch), 'p50_ms': timing['p50_ms'],
                    'p95_ms': timing['p95_ms'], 'texts_per_second': timing['texts_per_second'],
                    'retrieval_recall_at_1': variant['quality']['recall_at_1']})
        st.dataframe(rows, hide_index=True)
        st.caption('Warm forward calls; tokenization, queueing and HTTP excluded. Two CPU inference threads. Retrieval set: 24 authored queries/documents.')
        with st.expander('Actual HTTP measurements and error rates'):
            rows = []
            for config in report['http']:
                for load in config['loads']:
                    rows.append({'backend': config['backend'], 'max_batch': config['max_batch'], 'concurrency': load['concurrency'],
                        'p95_success_ms': load['successful_latency']['p95_ms'], 'success_rps': load['successful_requests_per_second'],
                        'error_rate': load['error_rate']})
            st.dataframe(rows, hide_index=True)
            st.caption('Loopback HTTP, closed-loop load, warm-up excluded. See raw report for all attempts, failures and batch sizes.')
    st.subheader('Try semantic search locally')
    backend = st.selectbox('Backend', ['onnx-int8', 'onnx-fp32', 'torch-fp32'])
    text = st.text_area('English search query', 'How do I stop automatic monthly renewals?', max_chars=2000)
    if st.button('Search local documents'):
        try:
            with st.spinner('Loading and running the actual local encoder...'):
                engine, vectors, lock = load_search(backend)
                with lock:
                    start = time.perf_counter()
                    query = engine.encode([text])[0]
                    scores = vectors @ query
                    elapsed = (time.perf_counter() - start) * 1000
            order = np.argsort(-scores, kind='stable')[:3]
            st.dataframe([{'document': corpus()[int(i)]['id'], 'cosine_score': float(scores[i]), 'text': corpus()[int(i)]['text']} for i in order], hide_index=True)
            st.caption(f'CPU local search: {elapsed:.2f} ms, excluding model/corpus load. This UI runs the engine directly; it is not an HTTP load test. Cosine scores are not probabilities.')
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            import logging
            logging.exception('ServeBench local search error')
            st.error('Unable to load/run local artifacts. Run download_model.py and export_model.py first; inspect the console.')


if __name__ == '__main__':
    main()
