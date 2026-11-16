import unittest
import numpy as np
from fastapi.testclient import TestClient
from api import create_app


class FakeEngine:
    def prepare(self, texts):
        if any(len(t.split()) > 128 for t in texts):
            raise ValueError('Over token limit')

    def encode(self, texts):
        return np.ones((len(texts), 384), dtype=np.float32)


class ApiTests(unittest.TestCase):
    def test_ready_and_embedding_contract_with_injected_engine(self):
        with TestClient(create_app(FakeEngine())) as client:
            self.assertTrue(client.get('/health').json()['ready'])
            result = client.post('/embed', json={'text': 'hello'})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(len(result.json()['embedding']), 384)
            self.assertEqual(client.get('/metrics').json()['completed'], 1)

    def test_rejects_blank_extra_fields_and_invalid_deadline(self):
        with TestClient(create_app(FakeEngine())) as client:
            for body in ({'text': ' '}, {'text': 'hi', 'timeout_ms': 0}, {'text': 'hi', 'unknown': 1}, {'text': 'x' * 2001}):
                self.assertEqual(client.post('/embed', json=body).status_code, 422)

    def test_model_token_limit_is_422(self):
        with TestClient(create_app(FakeEngine())) as client:
            self.assertEqual(client.post('/embed', json={'text': 'word ' * 200}).status_code, 422)
