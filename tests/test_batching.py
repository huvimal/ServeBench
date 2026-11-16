import asyncio
import time
import unittest
from batching import Batcher, Overloaded, DeadlineExceeded


class BatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_batches_preserve_result_order_and_flush_partial(self):
        calls = []
        def infer(texts):
            calls.append(texts)
            return [[len(t)] for t in texts]
        batcher = Batcher(infer, capacity=16, max_batch=4, wait_ms=5)
        await batcher.start()
        try:
            results = await asyncio.gather(*(batcher.submit(t, 1000) for t in ['a', 'bb', 'ccc', 'dddd', 'eeeee']))
            self.assertEqual([r['embedding'] for r in results], [[1], [2], [3], [4], [5]])
            self.assertEqual([len(c) for c in calls], [4, 1])
        finally:
            await batcher.close()

    async def test_queue_rejects_without_unbounded_wait(self):
        def infer(texts):
            time.sleep(.08)
            return [[1] for _ in texts]
        batcher = Batcher(infer, capacity=1, max_batch=1, wait_ms=0)
        await batcher.start()
        try:
            first = asyncio.create_task(batcher.submit('first', 1000))
            await asyncio.sleep(.02)
            second = asyncio.create_task(batcher.submit('second', 1000))
            await asyncio.sleep(.005)
            with self.assertRaises(Overloaded):
                await batcher.submit('third', 1000)
            await asyncio.gather(first, second)
            self.assertEqual(batcher.stats['rejected'], 1)
            self.assertLessEqual(batcher.stats['max_queue_seen'], 1)
        finally:
            await batcher.close()

    async def test_timeout_does_not_spawn_an_extra_inference_worker(self):
        active, peak = 0, 0
        def infer(texts):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            time.sleep(.04)
            active -= 1
            return [[1] for _ in texts]
        batcher = Batcher(infer, max_batch=1, wait_ms=0)
        await batcher.start()
        try:
            with self.assertRaises(DeadlineExceeded):
                await batcher.submit('late', 1)
            await batcher.submit('next', 1000)
            self.assertEqual(peak, 1)
            self.assertEqual(batcher.stats['timed_out'], 1)
        finally:
            await batcher.close()

    async def test_one_bad_result_does_not_fail_its_neighbor(self):
        batcher = Batcher(lambda texts: [ValueError('bad') if t == 'bad' else [7] for t in texts], max_batch=8)
        await batcher.start()
        try:
            results = await asyncio.gather(batcher.submit('bad', 1000), batcher.submit('good', 1000), return_exceptions=True)
            self.assertIsInstance(results[0], ValueError)
            self.assertEqual(results[1]['embedding'], [7])
        finally:
            await batcher.close()

    async def test_inference_failure_is_reported_and_worker_recovers(self):
        def infer(texts):
            if texts[0] == 'fail':
                raise RuntimeError('test failure')
            return [[3]]
        batcher = Batcher(infer, max_batch=1, wait_ms=0)
        await batcher.start()
        try:
            with self.assertRaises(RuntimeError):
                await batcher.submit('fail', 1000)
            self.assertEqual((await batcher.submit('ok', 1000))['embedding'], [3])
        finally:
            await batcher.close()

    async def test_cancelled_request_does_not_break_next_request(self):
        batcher = Batcher(lambda texts: [[9] for _ in texts], max_batch=8, wait_ms=20)
        await batcher.start()
        try:
            pending = asyncio.create_task(batcher.submit('cancel', 1000))
            await asyncio.sleep(.001)
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
            self.assertEqual((await batcher.submit('ok', 1000))['embedding'], [9])
        finally:
            await batcher.close()
