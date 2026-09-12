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
    provides the toolchain, via its own `--require-*` flag). A kernel that loaded
    and then refused its arguments is a different thing and raises
    `KernelRejected`: that is never a skip, because no toolchain would make it
    go away.
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


class KernelRejected(ValueError):
    """A kernel that loaded and ran refused its arguments.

    Separate from `BackendUnavailable` because the two mean opposite things to a
    caller. Unavailability is about the *host* -- no compiler, no `bcir`, a build
    failure -- and is an honest skip on a laptop. A rejection is about the *call*:
    the kernels are here, they ran, and they said no, which is a defect in the
    caller or in the data it passed.

    All three doors used to raise `BackendUnavailable` for both. Asking for a
    `top_k` above the kernel's own `BCIR_AI_MAX_TOP_K` therefore printed
    `[skip] native backend unavailable` and exited 0, and `--backend both` reported
    no disagreement because it had run no differential to disagree in. A skip that
    can be produced by an argument is not a statement about the host
    (`docs/security/laws.md` L1: the label is part of the verdict).
    """


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
        raise KernelRejected(f"BCIR Q8 quantization failed: {exc}") from exc


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
        raise KernelRejected(f"BCIR Q8 rows-dot failed: {exc}") from exc


def eligibility_mask(rows: int, admitted=None) -> bytes:
    """The kernel's `eligible` buffer: one byte per row, 1 admits and 0 skips.

    `admitted` of None is every row -- spelled as an explicit full mask rather than
    left to how a zero-length buffer happens to be interpreted.

    The representation is deliberately a dense mask over original row numbers rather
    than a compacted list of survivors. Compaction would renumber rows, and the
    reference and native backends are compared for *bit* equality on `(row, distance)`
    pairs; a renumbering would break that differential to save a few kilobytes at a
    corpus size where the whole mask is smaller than one vector.

    A row outside the set is a refusal, not a silently dropped entry: admitting a row
    that does not exist is a question the caller got wrong, and answering it anyway
    is how a filter comes to return the wrong rows quietly.
    """
    if admitted is None:
        return b"\x01" * rows
    mask = bytearray(rows)
    for row in admitted:
        index = int(row)
        if not 0 <= index < rows:
            raise ValueError(f"eligible row {index} is outside the set's {rows} rows")
        mask[index] = 1
    return bytes(mask)


def q15_topk(
    query: array,
    codes: array,
    *,
    rows: int,
    dim: int,
    top_k: int,
    eligible=None,
    build_dir=None,
):
    """Exact integer squared-L2 top-k, through `bcir_ai_q15_topk`.

    `eligible` is the mask the kernel has always documented and this door used to
    discard, hard-coding every row as a candidate. Passing it means a predicate is
    applied *inside* the scan -- a masked row never has its dot product computed --
    rather than by ranking everything and discarding afterwards.

    The kernel returns at most `top_k` matches and fewer when the mask admits fewer,
    so the result length is a fact about the query, not a guarantee. Callers size
    their output from what comes back.
    """
    kernels = load_kernels(build_dir)
    mask = eligibility_mask(rows, eligible)
    try:
        matches = kernels._q15_topk(array("h", query), codes, mask, rows, dim, top_k)
    except (ValueError, RuntimeError) as exc:
        raise KernelRejected(f"BCIR Q15 top-k failed: {exc}") from exc
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
