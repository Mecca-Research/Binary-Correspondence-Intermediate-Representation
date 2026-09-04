"""Small, bounded Python-oracle and native-C kernels for model-plan calibration.

Both engines are portable CPU floors and CI mechanisms; GPU/vendor measurements must
be supplied as separate ``KernelBenchmark`` records from the actual target channel.
Dimensions, repeats, subprocess time, and work are hard-bounded so this helper cannot
become a large inference job.
"""

from __future__ import annotations

import os
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from ..._artifact_json import strict_json_loads
from ...toolchain import host_link_args
from .assessment import KernelBenchmark

_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME = _ROOT / "runtime" / "c"


def _run_samples(work, repeats: int) -> tuple[int, int, int]:
    work()  # one bounded warmup
    samples = []
    checksum = 0.0
    for _ in range(repeats):
        start = time.perf_counter_ns()
        checksum += work()
        elapsed = time.perf_counter_ns() - start
        samples.append(max(1, elapsed))
    if not (-float("inf") < checksum < float("inf")):
        raise RuntimeError("model microbenchmark produced a non-finite checksum")
    ordered = sorted(samples)
    return ordered[0], int(statistics.median(ordered)), ordered[-1]


def run_bounded_model_microbench(
    *, channel: str = "host", weight_format: str = "source", dimension: int = 24, repeats: int = 5
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
        return dense_weights[index] if weight_format == "source" else q8_weights[index] * 0.0625

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

    pre = _run_samples(prefill, repeats)
    dec = _run_samples(decode, repeats)
    element_bytes = 4 if weight_format == "source" else 1
    return (
        KernelBenchmark(
            "prefill",
            channel,
            weight_format,
            *pre,
            n,
            2 * n * n * n,
            n * n * element_bytes,
            2 * n * n * 4,
            repeats,
        ),
        KernelBenchmark(
            "decode",
            channel,
            weight_format,
            *dec,
            1,
            2 * n * n,
            n * n * element_bytes,
            2 * n * 4,
            repeats,
        ),
    )


def run_bounded_model_microbench_native(
    *,
    channel: str = "host",
    weight_format: str = "source",
    dimension: int = 24,
    repeats: int = 5,
    cc: str | None = None,
) -> tuple[KernelBenchmark, KernelBenchmark]:
    """Compile and run the portable C measurement twin with no Python fallback."""
    if not isinstance(channel, str) or not channel:
        raise ValueError("microbenchmark channel must be nonempty")
    if weight_format not in ("source", "bcirq8-group32"):
        raise ValueError("microbenchmark format must be source or bcirq8-group32")
    if type(dimension) is not int or not 4 <= dimension <= 64:
        raise ValueError("microbenchmark dimension must be in [4, 64]")
    if type(repeats) is not int or not 3 <= repeats <= 15:
        raise ValueError("microbenchmark repeats must be in [3, 15]")
    compiler = (
        cc
        or os.environ.get("CC")
        or shutil.which("clang")
        or shutil.which("cc")
        or shutil.which("gcc")
    )
    sources = (_RUNTIME / "bcir_ai_kernels.c", _RUNTIME / "bcir_ai_microbench.c")
    if not compiler or any(not source.is_file() for source in sources):
        raise RuntimeError(
            "native model microbenchmark requires a C11 compiler and runtime sources"
        )
    with tempfile.TemporaryDirectory() as directory:
        executable = Path(directory) / (
            "bcir-ai-microbench.exe" if os.name == "nt" else "bcir-ai-microbench"
        )
        command = [
            compiler,
            "-std=c11",
            "-O3",
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-I",
            str(_RUNTIME),
            *(str(source) for source in sources),
            "-o",
            str(executable),
            *host_link_args(["-lm"]),
        ]
        build = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if build.returncode:
            raise RuntimeError(f"native model microbenchmark build failed:\n{build.stderr}")
        run = subprocess.run(
            [str(executable), str(dimension), str(repeats), weight_format],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if run.returncode:
            raise RuntimeError(f"native model microbenchmark failed:\n{run.stderr}")
    document = strict_json_loads(run.stdout, "native model microbenchmark", max_bytes=64 * 1024)
    fields = {"schema", "dimension", "repeats", "weight_format", "prefill", "decode"}
    sample_fields = {"lower_ns", "median_ns", "upper_ns", "checksum"}
    if (
        not isinstance(document, dict)
        or set(document) != fields
        or document["schema"] != "bcir.native_model_microbench.v1"
        or document["dimension"] != dimension
        or document["repeats"] != repeats
        or document["weight_format"] != weight_format
    ):
        raise RuntimeError("native model microbenchmark returned a malformed envelope")
    for operation in ("prefill", "decode"):
        row = document[operation]
        if (
            not isinstance(row, dict)
            or set(row) != sample_fields
            or any(
                type(row[field]) is not int or row[field] < 1
                for field in ("lower_ns", "median_ns", "upper_ns")
            )
            or isinstance(row["checksum"], bool)
            or not isinstance(row["checksum"], (int, float))
            or not -float("inf") < float(row["checksum"]) < float("inf")
        ):
            raise RuntimeError("native model microbenchmark returned a malformed sample")
    n = dimension
    element_bytes = 4 if weight_format == "source" else 1

    def result(operation: str) -> KernelBenchmark:
        row = document[operation]
        tokens = n if operation == "prefill" else 1
        operations = 2 * n * n * n if operation == "prefill" else 2 * n * n
        workspace = 2 * n * n * 4 if operation == "prefill" else 2 * n * 4
        return KernelBenchmark(
            operation,
            channel,
            weight_format,
            row["lower_ns"],
            row["median_ns"],
            row["upper_ns"],
            tokens,
            operations,
            n * n * element_bytes,
            workspace,
            repeats,
        )

    return result("prefill"), result("decode")


__all__ = ["run_bounded_model_microbench", "run_bounded_model_microbench_native"]
