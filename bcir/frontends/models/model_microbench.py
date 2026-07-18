"""Small, bounded reference kernels for model-plan calibration.

These kernels intentionally measure only the local Python reference path.  They are
useful as a portable floor and CI mechanism; GPU/vendor measurements must be supplied
as separate ``KernelBenchmark`` records from the actual target channel.  Dimensions,
repeats, and work are hard-bounded so this helper cannot become a large inference job.
"""

from __future__ import annotations

import statistics
import time

from .assessment import KernelBenchmark


def _run_samples(work, repeats: int) -> tuple[int, int, int]:
    work()  # one bounded warmup
    samples = []
    checksum = 0.0
    for _ in range(repeats):
        start = time.perf_counter_ns(); checksum += work(); elapsed = time.perf_counter_ns() - start
        samples.append(max(1, elapsed))
    if not (-float("inf") < checksum < float("inf")):
        raise RuntimeError("model microbenchmark produced a non-finite checksum")
    ordered = sorted(samples)
    return ordered[0], int(statistics.median(ordered)), ordered[-1]


def run_bounded_model_microbench(*, channel: str = "host", weight_format: str = "source",
                                 dimension: int = 24, repeats: int = 5
                                 ) -> tuple[KernelBenchmark, KernelBenchmark]:
    """Measure bounded prefill-like matmul and decode-like matvec reference kernels."""
    if not isinstance(channel, str) or not channel:
        raise ValueError("microbenchmark channel must be nonempty")
    if weight_format not in ("source", "bcirq8-group32"):
        raise ValueError("microbenchmark format must be source or bcirq8-group32")
    if type(dimension) is not int or not 4 <= dimension <= 64:
        raise ValueError("microbenchmark dimension must be in [4, 64]")
    if type(repeats) is not int or not 3 <= repeats <= 15:
        raise ValueError("microbenchmark repeats must be in [3, 15]")
    n = dimension
    activation = [((index * 17) % 31 - 15) / 16.0 for index in range(n * n)]
    dense_weights = [((index * 13) % 29 - 14) / 16.0 for index in range(n * n)]
    q8_weights = [((index * 13) % 127) - 63 for index in range(n * n)]

    def weight(index: int) -> float:
        return dense_weights[index] if weight_format == "source" \
            else q8_weights[index] * 0.0625

    def prefill() -> float:
        checksum = 0.0
        for i in range(n):
            row = i * n
            for j in range(n):
                value = 0.0
                for k in range(n):
                    value += activation[row + k] * weight(k * n + j)
                checksum += value
        return checksum

    vector = activation[:n]

    def decode() -> float:
        checksum = 0.0
        for j in range(n):
            value = 0.0
            for k in range(n):
                value += vector[k] * weight(k * n + j)
            checksum += value
        return checksum

    pre = _run_samples(prefill, repeats); dec = _run_samples(decode, repeats)
    element_bytes = 4 if weight_format == "source" else 1
    return (
        KernelBenchmark("prefill", channel, weight_format, *pre, n, 2 * n * n * n,
                        n * n * element_bytes, 2 * n * n * 4, repeats),
        KernelBenchmark("decode", channel, weight_format, *dec, 1, 2 * n * n,
                        n * n * element_bytes, 2 * n * 4, repeats),
    )


__all__ = ["run_bounded_model_microbench"]
