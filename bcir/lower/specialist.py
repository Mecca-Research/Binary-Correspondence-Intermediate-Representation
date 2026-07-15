"""JIT "specialist" synthesis: shape-baked, loop-unrolled kernels for hot shapes.

A generic kernel takes the trip count `n` at runtime, so the compiler must emit a
loop with a remainder branch and cannot fully unroll. When one concrete shape
recurs (telemetry: "this matmul is always 128x64"), BCIR synthesizes a
**specialist**: the count is baked as a compile-time constant and the body is
unrolled, so the resident compiler fully unrolls a constant-trip loop with no
remainder -- a hard-coded binary for exactly that shape, hot-swapped in for the
generic one.

GAINS-ONLY: a specialist is synthesized only when the shape is (a) *frequent* (seen
>= a threshold -- specializing a one-off wastes compile time and i-cache) and (b)
*small enough* to unroll (a huge baked count bloats code with no benefit, so large
shapes keep the generic runtime-`n` kernel). `synthesize()` returns the generic
kernel unchanged when neither holds, so the simple/large case is never penalized.
The specialist computes exactly the generic result (a correct specialization).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import Module
from ..kbcir.realize import RealizationResult
from .c_kernel import C_OP, _ctype
from .llvm import find_elementwise

# A shape is "frequent" once observed this many times; specialize only up to this
# many elements (above it, the generic runtime-n loop is better -- no code bloat).
DEFAULT_HOT_THRESHOLD = 8
DEFAULT_MAX_UNROLL = 4096
DEFAULT_UNROLL = 8


@dataclass(frozen=True)
class ShapeKey:
    op: str
    count: int
    elem: str = "f32"


@dataclass
class ShapeLedger:
    """Counts how often each (op, shape) is requested -- the hot-shape detector."""

    counts: dict[ShapeKey, int] = field(default_factory=dict)

    def observe(self, op: str, count: int, elem: str = "f32") -> ShapeKey:
        k = ShapeKey(op=op, count=count, elem=elem)
        self.counts[k] = self.counts.get(k, 0) + 1
        return k

    def frequency(self, key: ShapeKey) -> int:
        return self.counts.get(key, 0)

    def is_hot(self, key: ShapeKey, threshold: int = DEFAULT_HOT_THRESHOLD) -> bool:
        return self.frequency(key) >= threshold

    def hot_shapes(self, threshold: int = DEFAULT_HOT_THRESHOLD) -> list[ShapeKey]:
        return sorted((k for k, n in self.counts.items() if n >= threshold),
                      key=lambda k: (k.op, k.count))


def specializable(count: int, max_unroll: int = DEFAULT_MAX_UNROLL) -> bool:
    """A shape is worth baking only if it is small enough to unroll without bloat."""
    return 1 <= count <= max_unroll


def emit_specialist_c(op_str: str, c_op: str, count: int, elem: str = "f32",
                      fn_name: str | None = None, unroll: int = DEFAULT_UNROLL) -> str:
    """Emit a shape-baked, unrolled specialist for `C = A op B` over exactly `count`
    elements. The count is a compile-time constant (no `n` parameter), the inner
    block is unrolled by `unroll`, and the `count % unroll` remainder is baked as
    explicit statements -- so the kernel is correct for this exact shape and the
    compiler fully unrolls a constant-trip loop."""
    ctype = _ctype(elem)
    fn = fn_name or f"bcir_kernel_{op_str.replace('.', '_')}_{count}"
    u = max(1, min(unroll, count))
    full = (count // u) * u
    inc = "#include <stddef.h>\n" + ("#include <stdint.h>\n" if elem == "i32" else "")
    fp = "" if elem == "i32" else "#pragma STDC FP_CONTRACT OFF\n"
    block = "  ".join(f"C[i + {j}u] = A[i + {j}u] {c_op} B[i + {j}u];" for j in range(u))
    body = [f"void {fn}(const {ctype} *restrict A, const {ctype} *restrict B,",
            f"             {ctype} *restrict C) {{",
            f"  /* shape specialist: n = {count} baked constant, unroll {u} */"]
    if full > 0:
        body.append(f"  for (size_t i = 0; i + {u}u <= {full}u; i += {u}u) {{ {block} }}")
    for r in range(full, count):                       # baked remainder (count % unroll)
        body.append(f"  C[{r}u] = A[{r}u] {c_op} B[{r}u];")
    body.append("}")
    return (f"/* BCIR -> shape specialist (n={count}, op={op_str}, elem={ctype}; "
            f"hot-shape JIT). */\n{inc}"
            f'_Static_assert(sizeof({ctype}) == 4, "BCIR {ctype} specialist needs a 4-byte element");\n'
            f"{fp}\n" + "\n".join(body) + "\n")


@dataclass(frozen=True)
class SpecialistResult:
    specialized: bool
    fn_name: str
    count: int
    kernel_c: str
    reason: str


def synthesize(module: Module, result: RealizationResult, ledger: ShapeLedger,
               elem: str = "f32", *, threshold: int = DEFAULT_HOT_THRESHOLD,
               max_unroll: int = DEFAULT_MAX_UNROLL, hw_width: int | None = None) -> SpecialistResult:
    """Pick a kernel for the module's elementwise claim: a shape-baked specialist if
    its shape is hot and small, else the generic kernel (`emit_kernel_c`). Records
    the request in `ledger` first, so frequency reflects this call too."""
    claim, _ = find_elementwise(module, result)
    key = ledger.observe(claim.op or "vector.op", max(1, claim.count), elem)
    c_op = C_OP.get(claim.opcode, "+")
    if ledger.is_hot(key, threshold) and specializable(key.count, max_unroll):
        fn = f"bcir_kernel_{key.op.replace('.', '_')}_{key.count}"
        return SpecialistResult(
            specialized=True, fn_name=fn, count=key.count,
            kernel_c=emit_specialist_c(key.op, c_op, key.count, elem, fn),
            reason=f"hot shape n={key.count} (x{ledger.frequency(key)}) -> baked specialist")
    # generic fallback (the gains-only no-op: cold or large shapes are not baked).
    from .c_kernel import emit_kernel_c
    reason = ("large shape: generic runtime-n loop" if not specializable(key.count, max_unroll)
              else f"shape not yet hot (x{ledger.frequency(key)} < {threshold})")
    return SpecialistResult(
        specialized=False, fn_name="bcir_kernel", count=key.count,
        kernel_c=emit_kernel_c(module, result, "bcir_kernel", elem, hw_width=hw_width),
        reason=reason)
