import os
import platform
import statistics
import threading
import time
import psutil


def summary_ms(values):
    if not values:
        return {'n': 0, 'p50_ms': None, 'p95_ms': None, 'mean_ms': None}
    ordered = sorted(values)
    import math
    return {'n': len(values), 'p50_ms': statistics.median(values),
            'p95_ms': ordered[math.ceil(.95 * len(values)) - 1], 'mean_ms': statistics.mean(values)}


class MemorySampler:
    def __init__(self, pid=None):
        self.process = psutil.Process(pid or os.getpid())
        self.peak = self.process.memory_info().rss
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop_event.wait(.01):
            try:
                self.peak = max(self.peak, self.process.memory_info().rss)
            except psutil.NoSuchProcess:
                return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop_event.set()
        self.thread.join()


def hardware():
    return {'platform': platform.platform(), 'cpu': os.getenv('PROCESSOR_IDENTIFIER') or platform.processor(),
        'logical_cpus': psutil.cpu_count(), 'physical_cores': psutil.cpu_count(logical=False),
        'ram_bytes': psutil.virtual_memory().total, 'python': platform.python_version(),
        'device': 'cpu', 'scope': 'One local machine; background tasks, power and thermal state are not controlled.'}
