"""Loopback CPU embedding API. A fixed backend per process keeps measurements interpretable."""
from contextlib import asynccontextmanager
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from batching import Batcher, Overloaded, DeadlineExceeded


class EmbedRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: str = Field(min_length=1, max_length=2000)
    timeout_ms: int = Field(default=2000, ge=1, le=10000)

    @field_validator('text')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Text cannot be blank')
        return value


def create_app(engine=None, capacity=None, max_batch=None, wait_ms=None):
    backend = os.getenv('SERVEBENCH_BACKEND', 'onnx-int8')

    @asynccontextmanager
    async def lifespan(app):
        nonlocal engine
        if engine is None:
            from engine import Engine
            engine = Engine(backend, int(os.getenv('SERVEBENCH_THREADS', '2')))
        def infer_many(texts):
            # Validate per item in the same worker so one overlength item cannot poison its neighbors.
            valid, positions = [], []
            results = [None] * len(texts)
            for i, text in enumerate(texts):
                try:
                    engine.prepare([text])
                    valid.append(text)
                    positions.append(i)
                except ValueError as exc:
                    results[i] = exc
            if valid:
                for i, vector in zip(positions, engine.encode(valid)):
                    results[i] = vector
            return results
        batcher = Batcher(infer_many, capacity if capacity is not None else int(os.getenv('SERVEBENCH_QUEUE', '16')),
            max_batch if max_batch is not None else int(os.getenv('SERVEBENCH_BATCH', '8')),
            wait_ms if wait_ms is not None else float(os.getenv('SERVEBENCH_WAIT_MS', '5')))
        app.state.batcher = batcher
        await batcher.start()
        try:
            yield
        finally:
            await batcher.close()

    app = FastAPI(title='ServeBench', lifespan=lifespan)

    @app.get('/health')
    async def health():
        return {'ready': hasattr(app.state, 'batcher'), 'backend': backend, 'device': 'cpu', 'pid': os.getpid()}

    @app.get('/metrics')
    async def metrics():
        return app.state.batcher.snapshot()

    @app.post('/embed')
    async def embed(request: EmbedRequest):
        try:
            result = await app.state.batcher.submit(request.text, request.timeout_ms)
            return {'backend': backend, 'device': 'cpu', **result}
        except Overloaded as exc:
            raise HTTPException(429, str(exc), headers={'Retry-After': '1'}) from None
        except DeadlineExceeded as exc:
            raise HTTPException(504, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        except Exception:
            logging.exception('Local model inference failed')
            raise HTTPException(500, 'Local inference failed; inspect server log') from None
    return app


app = create_app()
