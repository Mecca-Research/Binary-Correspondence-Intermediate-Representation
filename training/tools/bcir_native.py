#!/usr/bin/env python3
"""The corpus's one door into BCIR.

`runtime/c/bcir_ai_kernels.c` carries the arithmetic this corpus needs and BCIR
already gates: a symmetric power-of-two quantizer, an exact-integer top-k, and
an output-major dot kernel whose own header names embedding projections as its
use. Calling those beats writing a second set, and it means the vectors this
corpus publishes are BCIR artifacts rather than lookalikes.

Every tool here reaches them through this module and no other way, so the rules
below are stated once instead of per caller:

  * **The import is lazy.** Nothing in `training/` imports BCIR at module scope.
    `training/` is never a build dependency of BCIR, and this does not make BCIR
    a dependency of `training/` either -- every tool that calls in here has a
    pure-Python path that works when this one is unavailable.
  * **Unavailability is a skip that says so.** No compiler, no `bcir` package, a
    build failure: all raise `BackendUnavailable`, and each caller decides
    whether that is expected here (a laptop) or a defect (the CI job that
    provides the toolchain, via its own `--require-*` flag).
  * **The kernels are never a fallback.** A caller asks for native or asks for
    reference. Silently substituting one for the other would make a differential
    between them meaningless, since it could compare a thing to itself.

Two doors live here, under the same rules: the native C kernels above, and
BCIR's hosted training contracts below. The second is what stops `training/`
from being a passive database -- the corpus feeds BCIR's own corpus
preparation, tokenizer, example contracts, and pipeline ledger instead of
emitting JSON that nothing in this repository consumes.
"""

from __future__ import annotations

from array import array
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
CORPUS_ROOT = TOOLS_DIR.parent
REPO_ROOT = CORPUS_ROOT.parent

DEFAULT_BUILD_DIR = Path("build/training/native")

# BCIRQ8 groups share one power-of-two exponent. 64 divides the corpus's vector
# dimension, so a group never straddles two rows -- one row's dynamic range
# cannot then set the grid for its neighbour's coordinates.
DEFAULT_GROUP_SIZE = 64


class BackendUnavailable(RuntimeError):
    """BCIR's native rail cannot be reached here. Honest skip, never a switch."""


_CACHE: dict[Path, object] = {}


def load_kernels(build_dir: Path | None = None):
    """Build and load `bcir_ai_kernels`, or raise `BackendUnavailable`.

    Cached per build directory: the gate calls this once per probe and the
    library costs a compiler invocation to produce.
    """
    import sys

    target = Path(build_dir or DEFAULT_BUILD_DIR)
    if target in _CACHE:
        return _CACHE[target]

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from bcir.kbcir.native_ai import NativeAIKernels
    except ImportError as exc:
        raise BackendUnavailable(f"bcir.kbcir.native_ai is not importable: {exc}") from exc

    try:
        kernels = NativeAIKernels.build(target)
    except RuntimeError as exc:
        raise BackendUnavailable(f"could not build BCIR's native AI kernels: {exc}") from exc

    _CACHE[target] = kernels
    return kernels


def quantize_q8(values: list[float], *, group_size: int = DEFAULT_GROUP_SIZE, build_dir=None):
    """Quantize with BCIR's own bridge, not a reimplementation of it.

    Returns the library's `NativeQuantizedTensor`: per-group power-of-two
    exponents plus signed 8-bit codes, the format BCIR ships model weights in.
    Its constructor rejects the asymmetric code -128, so the symmetry discipline
    is enforced by BCIR rather than restated here.
    """
    kernels = load_kernels(build_dir)
    try:
        return kernels.quantize(values, group_size=group_size, bits=8)
    except (ValueError, RuntimeError) as exc:
        raise BackendUnavailable(f"BCIR Q8 quantization failed: {exc}") from exc


def q8_rows_dot(query: list[float], tensor, *, rows: int, build_dir=None) -> tuple[float, ...]:
    """One dot product per row, through `bcir_ai_q8_rows_dot_f64`.

    That kernel's header describes it as the output-major counterpart "used by
    embedding/head projections", which is exactly this: a [rows, dim] matrix of
    quantized corpus vectors against one query. Because the stored vectors are
    unit length, each dot IS a cosine, so ranking by it descending is a cosine
    ranking -- no separate normalization pass.
    """
    kernels = load_kernels(build_dir)
    try:
        return kernels.q8_rows_dot(
            query,
            tensor.codes,
            tensor.exponents,
            output_count=rows,
            group_size=tensor.group_size,
        )
    except (ValueError, RuntimeError) as exc:
        raise BackendUnavailable(f"BCIR Q8 rows-dot failed: {exc}") from exc


def q15_topk(query: array, codes: array, *, rows: int, dim: int, top_k: int, build_dir=None):
    """Exact integer squared-L2 top-k, through `bcir_ai_q15_topk`.

    Every row is a candidate. The kernel documents `eligible` as optional, but an
    explicit full mask says so in the call rather than relying on how a
    zero-length buffer happens to be interpreted.
    """
    kernels = load_kernels(build_dir)
    try:
        matches = kernels._q15_topk(array("h", query), codes, b"\x01" * rows, rows, dim, top_k)
    except (ValueError, RuntimeError) as exc:
        raise BackendUnavailable(f"BCIR Q15 top-k failed: {exc}") from exc
    return [(index, distance) for index, distance in matches]


# --------------------------------------------------------------------------
# The hosted training stack
# --------------------------------------------------------------------------

# These modules are deliberately tensor-framework free -- BCIR's own contracts
# module says so in its first paragraph -- so the corpus can build real training
# examples, and CI can gate them, without torch anywhere near the runner.
HOSTED_MODULES = ("contracts", "data", "bpe", "pipeline")


def load_hosted_training(*names: str):
    """Import BCIR's hosted training modules, or raise `BackendUnavailable`.

    Same posture as the kernels: lazy, opt-in, and never silently replaced by a
    local reimplementation. If BCIR is not importable the caller reports an
    honest skip rather than quietly exporting examples built to a contract this
    corpus invented for itself -- which would defeat the point of using BCIR's.
    """
    import importlib
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    wanted = names or HOSTED_MODULES
    loaded = {}
    for name in wanted:
        try:
            loaded[name] = importlib.import_module(f"bcir.hosted.training.{name}")
        except ImportError as exc:
            raise BackendUnavailable(
                f"bcir.hosted.training.{name} is not importable: {exc}"
            ) from exc
    return loaded
