"""One inference worker, bounded admission queue, dynamic batching and deadlines."""
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


class Overloaded(Exception):
    pass


class DeadlineExceeded(Exception):
    pass


@dataclass
class Item:
    text: str
    deadline: float
    started: float
    future: asyncio.Future


class Batcher:
    def __init__(self, infer_many, capacity=16, max_batch=8, wait_ms=5):
        if not 1 <= capacity <= 256 or not 1 <= max_batch <= 32 or not 0 <= wait_ms <= 50:
            raise ValueError('Invalid bounded serving configuration')
        self.infer_many = infer_many
        self.queue = asyncio.Queue(maxsize=capacity)
        self.max_batch, self.wait_ms = max_batch, wait_ms
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='inference')
        self.task = None
        self.closing = False
        self.stats = dict(accepted=0, rejected=0, completed=0, timed_out=0, cancelled=0, failed=0,
                          batches=0, max_batch_seen=0, max_queue_seen=0)

    async def start(self):
        self.task = asyncio.create_task(self.run())

    async def submit(self, text, timeout_ms):
        if self.closing:
            raise Overloaded('Server is stopping')
        loop = asyncio.get_running_loop()
        now = time.perf_counter()
        item = Item(text, now + timeout_ms / 1000, now, loop.create_future())
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            self.stats['rejected'] += 1
            raise Overloaded('Admission queue is full') from None
        self.stats['accepted'] += 1
        self.stats['max_queue_seen'] = max(self.stats['max_queue_seen'], self.queue.qsize())
        try:
            return await asyncio.wait_for(item.future, max(0, item.deadline - time.perf_counter()))
        except (TimeoutError, DeadlineExceeded):
            self.stats['timed_out'] += 1
            raise DeadlineExceeded('Request deadline exceeded') from None
        except asyncio.CancelledError:
            item.future.cancel()
            self.stats['cancelled'] += 1
            raise

    async def run(self):
        loop = asyncio.get_running_loop()
        while True:
            first = await self.queue.get()
            items = [first]
            until = min(time.perf_counter() + self.wait_ms / 1000, first.deadline)
            while len(items) < self.max_batch:
                try:
                    items.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    remaining = until - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        items.append(await asyncio.wait_for(self.queue.get(), remaining))
                    except TimeoutError:
                        break
            live = []
            for item in items:
                if item.future.done():
                    continue
                if time.perf_counter() >= item.deadline:
                    item.future.set_exception(DeadlineExceeded())
                else:
                    live.append(item)
            if live:
                start = time.perf_counter()
                self.stats['batches'] += 1
                self.stats['max_batch_seen'] = max(self.stats['max_batch_seen'], len(live))
                try:
                    values = await loop.run_in_executor(self.pool, self.infer_many, [i.text for i in live])
                    if len(values) != len(live):
                        raise RuntimeError('Inference returned the wrong batch length')
                except Exception as exc:
                    values = [exc] * len(live)
                elapsed = time.perf_counter() - start
                for item, value in zip(live, values):
                    if item.future.done():
                        continue
                    if time.perf_counter() >= item.deadline:
                        item.future.set_exception(DeadlineExceeded())
                    elif isinstance(value, Exception):
                        self.stats['failed'] += 1
                        # Do not share a traceback from this live worker coroutine with a request.
                        # Consumers may clear exception frames; that must not terminate the worker.
                        clean_error = ValueError(str(value)) if isinstance(value, ValueError) else RuntimeError(str(value))
                        item.future.set_exception(clean_error)
                    else:
                        self.stats['completed'] += 1
                        item.future.set_result({'embedding': value.tolist() if hasattr(value, 'tolist') else value,
                            'batch_size': len(live), 'batch_inference_ms': elapsed * 1000,
                            'server_total_ms': (time.perf_counter() - item.started) * 1000})
            for _ in items:
                self.queue.task_done()

    async def close(self):
        self.closing = True
        await self.queue.join()
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass
        # A timed-out running call is not killed. The sole worker remains occupied until it returns.
        self.pool.shutdown(wait=True)

    def snapshot(self):
        return {**self.stats, 'queue_depth': self.queue.qsize(), 'queue_capacity': self.queue.maxsize,
                'max_batch': self.max_batch, 'batch_wait_ms': self.wait_ms, 'inference_workers': 1}
