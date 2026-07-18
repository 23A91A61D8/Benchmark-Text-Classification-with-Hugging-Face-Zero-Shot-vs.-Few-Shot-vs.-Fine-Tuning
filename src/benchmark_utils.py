"""Shared helpers for measuring latency and peak memory usage."""
import time
import statistics
from contextlib import contextmanager

import torch
import psutil
import os

from src import config

logger = config.get_logger(__name__)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@contextmanager
def track_peak_memory():
    """Context manager yielding a callable that returns peak memory in MB.

    Uses CUDA's peak allocator stats when a GPU is present; otherwise falls
    back to tracking the process's resident set size (RSS) delta, which is
    the best available proxy for peak memory usage on CPU-only hosts.
    """
    device = get_device()
    process = psutil.Process(os.getpid())

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_rss = None
    else:
        start_rss = process.memory_info().rss

    peak_holder = {"mb": 0.0}
    try:
        yield peak_holder
    finally:
        if device == "cuda":
            torch.cuda.synchronize()
            peak_holder["mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            end_rss = process.memory_info().rss
            peak_holder["mb"] = max(end_rss, start_rss) / (1024 ** 2)


def measure_latency(fn, n_warmup=None, n_runs=None):
    """Run `fn()` repeatedly and return (median_ms, p99_ms) over n_runs calls."""
    n_warmup = n_warmup if n_warmup is not None else config.LATENCY_WARMUP_RUNS
    n_runs = n_runs if n_runs is not None else config.LATENCY_MEASURED_RUNS
    device = get_device()

    for _ in range(n_warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()

    timings = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        timings.append((end - start) * 1000.0)

    timings.sort()
    median_ms = statistics.median(timings)
    p99_index = min(len(timings) - 1, int(round(0.99 * (len(timings) - 1))))
    p99_ms = timings[p99_index]
    return median_ms, p99_ms


def accuracy_and_macro_f1(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score

    return (
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="macro")),
    )
