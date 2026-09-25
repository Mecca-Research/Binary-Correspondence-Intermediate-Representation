"""The declared alias facts of the elementwise kernel: one derivation, every emitter and law.

BCIR does not infer aliasing. A claim DECLARES the resources it reads and writes (RIDs), its
hazard contract and its volatility, and every resource declares its element size -- strictly
more than an alias analysis could recover downstream. The kernel `C[i] = A[i] op B[i]` reaches
LLVM through the textual emitter (`lower.llvm`, which the AOT, JIT and WASM paths share) and
through the C emitters clang lowers (`lower.c_kernel`, the hot-shape `lower.specialist`), and
each carries the facts derived here, once (docs/security/laws.md L14: the four predicates this
replaces disagreed, and the specialist's wrote `restrict` on all three pointers of an in-place
claim):

  * **The RID partition** of the operand positions (A, B, C) = (rd[0], rd[1], wr[0]). A position
    whose resource no other position names is EXCLUSIVE, and only an exclusive pointer carries
    LLVM `noalias` or C `restrict`; each resource is one alias scope, every access carrying its
    own resource's scope in `!alias.scope` and every other resource's in `!noalias`. The scopes
    live in one domain per kernel, so an inliner clones them per call site: they relate the
    accesses of one execution and never assert anything about another.
  * **The element type**: the lowering contract's (`elem`), whose size every operand resource
    must declare. A 4-byte kernel over a resource of 8-byte elements addresses the wrong bytes,
    so it is refused, never emitted. Its TBAA tag is the one clang gives the same C type
    (`"float"` / `"int"` under `"Simple C/C++ TBAA"`), which is what makes the tag sound: the
    kernel is called through a C ABI with exactly those pointer types, so it asserts nothing C's
    effective-type rule does not already require of its caller -- and under LTO the kernel and
    its C caller share one type system instead of two roots LLVM cannot compare.
  * **Volatility**: every access of a volatile claim is volatile, and no access of any other.
  * **The hazard**: `unique` needs no ordering; `barriered` is a full barrier, a sequentially
    consistent fence before the kernel's first access and after its last. An `atomic` hazard
    needs atomic element operations and an unknown one names no ordering; the subset generates
    neither, so `find_elementwise` refuses both rather than lowering them to plain accesses.
"""

from __future__ import annotations

from dataclasses import dataclass

POSITIONS = ("A", "B", "C")

#: The element types the elementwise subset lowers, by lowering contract: the LLVM type, the
#: scalar type clang's TBAA names for the same C type, the C type, and its size in bytes.
ELEMENT_TYPES = {
    "f32": ("float", "float", "float", 4),
    "i32": ("i32", "int", "int32_t", 4),
}

#: The hazard contracts the subset realizes, and the fence ordering each needs (None: none).
HAZARD_FENCES = {"unique": None, "barriered": "seq_cst"}

#: The TBAA type DAG clang emits for C, which the kernel's accesses share.
TBAA_ROOT = "Simple C/C++ TBAA"
TBAA_CHAR = "omnipotent char"

#: The C11 fence a barriered kernel executes first and last (`<stdatomic.h>`).
C_FENCE = "atomic_thread_fence(memory_order_seq_cst);"


def hazard_refusal(hazard: str) -> str | None:
    """Why the elementwise subset cannot lower a claim with this hazard contract, or None."""
    if hazard in HAZARD_FENCES:
        return None
    if hazard == "atomic":
        return (
            "an atomic hazard needs atomic element operations, which the elementwise subset "
            "does not generate (it would lower them to plain loads and stores)"
        )
    return f"hazard {hazard!r} names no ordering the elementwise subset can realize"


@dataclass(frozen=True)
class KernelFacts:
    """What the claim declares about the kernel's three pointers and its element type."""

    rids: tuple  # the resource behind each position (A, B, C)
    resources: tuple  # the distinct RIDs, in order of first appearance over (A, B, C)
    exclusive: tuple  # per position: no other position names its resource
    volatile: bool
    fence: str | None  # the fence ordering a barriered claim needs; None for a unique one
    elem: str | None  # the lowering contract's element type; None when not derived
    ll_type: str | None
    tbaa: str | None
    ctype: str | None

    def others(self, position: int) -> tuple:
        """The resources a position's accesses provably do not touch."""
        return tuple(r for r in self.resources if r != self.rids[position])

    def c_param(self, position: int, ctype: str | None = None, *, tight: bool = False) -> str:
        """The C declaration of a pointer parameter: `const` for a read, `volatile` for a
        volatile claim, `restrict` exactly where the position is exclusive. `tight` is the
        header's spelling (`*restrict A`); the kernels write `* restrict A`."""
        const = "const " if position < 2 else ""
        vol = "volatile " if self.volatile else ""
        name = POSITIONS[position]
        if tight:
            qual = "restrict " if self.exclusive[position] else ""
            return f"{const}{vol}{ctype or self.ctype} *{qual}{name}"
        qual = " restrict" if self.exclusive[position] else ""
        return f"{const}{vol}{ctype or self.ctype} *{qual} {name}"


def kernel_facts(module, claim, elem: str | None = "f32") -> KernelFacts:
    """The declared facts of the elementwise claim's kernel. Raises NotImplementedError for what
    the subset does not lower: an element type it does not emit, an operand resource that is not
    declared or declares another element size. `elem=None` derives only the element-independent
    facts (the Q-fixed kernel, whose lanes are its own representation)."""
    rids = (claim.rd[0], claim.rd[1], claim.wr[0])
    resources = tuple(dict.fromkeys(rids))
    exclusive = tuple(sum(r == other for other in rids) == 1 for r in rids)
    refusal = hazard_refusal(claim.hazard)
    if refusal is not None:
        raise NotImplementedError(f"claim {claim.id}: {refusal}")
    ll_type = tbaa = ctype = None
    if elem is not None:
        if elem not in ELEMENT_TYPES:
            raise NotImplementedError(
                f"the elementwise lowering subset emits {sorted(ELEMENT_TYPES)} elements, "
                f"not {elem!r}"
            )
        ll_type, tbaa, ctype, size = ELEMENT_TYPES[elem]
        for rid in resources:
            resource = module.resources.get(rid)
            if resource is None:
                raise NotImplementedError(
                    f"claim {claim.id} names RID {rid}, which the module does not declare: the "
                    f"kernel has no declared element type to address it with"
                )
            if resource.elem_bytes != size:
                raise NotImplementedError(
                    f"claim {claim.id}: RID {rid} declares {resource.elem_bytes}-byte elements "
                    f"and the {elem} kernel addresses {size}-byte ones"
                )
    return KernelFacts(
        rids,
        resources,
        exclusive,
        bool(claim.volatile),
        HAZARD_FENCES[claim.hazard],
        elem,
        ll_type,
        tbaa,
        ctype,
    )


def c_prologue_epilogue(facts: KernelFacts, indent: str = "  ") -> tuple[str, str]:
    """The statements a C kernel body opens and closes with: the barrier's fences, or none."""
    if facts.fence is None:
        return "", ""
    note = "/* barriered: a full barrier before the first access and after the last */"
    return f"{indent}{C_FENCE} {note}\n", f"{indent}{C_FENCE}\n"


def c_includes(facts: KernelFacts) -> str:
    return "#include <stdatomic.h>\n" if facts.fence is not None else ""


def bound_harness(
    facts: KernelFacts,
    ctype: str,
    patterns: tuple,
    call: str,
    *,
    total: str,
    index: str,
    want: str,
    want_type: str | None = None,
) -> tuple[str, str]:
    """The body of a self-check that binds the operands AS DECLARED: one buffer per resource
    (`R0`, `R1`, ...) and a snapshot of each (`S0`, ...), so an in-place claim runs in place and a
    claim reading one resource twice reads one buffer twice. Returns (setup and call, checks).

    `patterns[p]` initializes the resource position p is the first to name; `call` is the kernel
    call with `{A}`, `{B}`, `{C}` standing for the bound buffers; `want` is element `i`'s expected
    value below `n` in terms of `{SA}` and `{SB}`, the snapshots of the buffers bound to A and B.
    The written buffer must hold `want` below `n` and its snapshot from `n` on (an unmasked tail
    writes past `n`); every other buffer must hold its snapshot everywhere, so a kernel writing
    through a read pointer fails here instead of passing on three private buffers. `want_type`
    is the type `want` is computed and compared in (default: the element type) -- the Q-fixed
    self-check's 64-bit reference, so a result that does not fit the lane fails rather than
    being narrowed until it agrees."""
    bufs = [f"R{k}" for k in range(len(facts.resources))]
    snaps = [f"S{k}" for k in range(len(facts.resources))]
    at = {rid: k for k, rid in enumerate(facts.resources)}
    bound = [bufs[at[r]] for r in facts.rids]
    decl = "\n".join(
        f"  {ctype} *{bufs[k]} = malloc(bytes), *{snaps[k]} = malloc(bytes);"
        for k in range(len(bufs))
    )
    null = " || ".join(f"!{b}" for pair in zip(bufs, snaps) for b in pair)
    init = " ".join(
        f"{bufs[at[rid]]}[i] = {patterns[facts.rids.index(rid)]};" for rid in facts.resources
    )
    copy = " ".join(f"memcpy({snaps[k]}, {bufs[k]}, bytes);" for k in range(len(bufs)))
    written = at[facts.rids[2]]
    wtype = want_type or ctype
    got = f"({wtype}){{buf}}[i]" if want_type else "{buf}[i]"
    sa, sb = snaps[at[facts.rids[0]]], snaps[at[facts.rids[1]]]
    setup = (
        f"  /* The operands bound as the claim declares them: A, B, C = {', '.join(bound)} "
        f"(RIDs {', '.join(str(r) for r in facts.rids)}). */\n"
        f"  {index} total = {total};\n"
        f"  size_t bytes = (size_t)total * sizeof({ctype});\n"
        f"{decl}\n"
        f"  if ({null}) return 2;\n"
        f"  for ({index} i = 0; i < total; i++) {{ {init} }}\n"
        f"  {copy}\n"
        f"  {call.format(A=bound[0], B=bound[1], C=bound[2])}\n"
    )
    checks = []
    for k, buf in enumerate(bufs):
        if k == written:
            checks.append(
                f"  for ({index} i = 0; i < total; i++) {{\n"
                f"    {wtype} want = i < n ? ({want.format(SA=sa, SB=sb)}) : {snaps[k]}[i];\n"
                f"    if ({got.format(buf=buf)} != want) {{\n"
                f'      if (i < n) printf("FAIL n=%ld at %ld: got %g want %g\\n", (long)n, '
                f"(long)i, (double){buf}[i], (double)want);\n"
                f'      else printf("FAIL n=%ld: wrote past n at %ld (an unmasked tail)\\n", '
                f"(long)n, (long)i);\n"
                f"      return 1;\n"
                f"    }}\n"
                f"  }}\n"
            )
        else:
            checks.append(
                f"  for ({index} i = 0; i < total; i++)\n"
                f"    if ({buf}[i] != {snaps[k]}[i]) {{\n"
                f'      printf("FAIL n=%ld: wrote read-only RID {facts.resources[k]} at %ld\\n", '
                f"(long)n, (long)i);\n"
                f"      return 1;\n"
                f"    }}\n"
            )
    free = " ".join(f"free({b}); free({s_});" for b, s_ in zip(bufs, snaps))
    return setup, "".join(checks) + f"  {free}\n"
