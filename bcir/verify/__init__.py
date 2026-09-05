"""BCIR verifier: the runnable LangRef laws R1-R13.

LLVM verifies IR structure; BCIR verifies execution truth. The laws attach to the
artifacts of the correspondence chain, one entry point per artifact:

    verify(module)                       R1-R8  module / claim laws, + EV1-EV3 (event phases)
    verify_plan(module, result, h, ...)  R8-R9  K_BCIR plan laws (scope-aware with h, theta,
                                                policy, budget: the offer, every step cost and
                                                the budget are re-derived, never trusted)
    verify_pack(module, pack)            R10-R11 GEM stream laws
    verify_lowering(module, result, ll)  R12    lowering-contract law
    verify_provenance(portfolio, ...)    R13    policy/table provenance law
    verify_smart_lowering(module, ...)   R14-16 CIM dispatch / DVFS clock / alloc tier
    verify_all(...)                      the whole chain

Mirrored by the MLIR `-bcir-verify` pass (docs/PARITY.md): the structurally
checkable form of each law runs on the dialect; this oracle is the conformance
reference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..model import (
    ATOMIC_OPCODES,
    ISOLATED_DOMAINS,
    Claim,
    Domain,
    Lane,
    Module,
    Opcode,
    StrideClass,
    phase_graph_has_cycle,
    topological_phase_ids,
)


@dataclass(frozen=True)
class Diagnostic:
    law: str
    message: str


# Which lanes are legal for a declared access-pattern shape (LangRef R6).
# Lane A (atomic) is legal for SCALAR (a single-location atomic counter -- the canonical RMW,
# not on the decoupled GGG/scatter tail) as well as RANDOM (a scatter-atomic histogram update).
_LEGAL_LANES = {
    StrideClass.SCALAR: {Lane.U, Lane.H, Lane.A},
    StrideClass.UNIT: {Lane.U},
    StrideClass.STRIDED: {Lane.U, Lane.GGG},
    StrideClass.CACHELINE: {Lane.UX, Lane.GGG},
    StrideClass.TILE: {Lane.T},
    StrideClass.RANDOM: {Lane.GGG, Lane.A},
}

# Contract mnemonics (LangRef Sec. 5; mirror BCIRAttrs.td).
_HAZARDS = {"unique", "atomic", "barriered"}
_BOUNDS = {"strict", "masked", "assumed_safe"}
_VERIFY = {"none", "bounds", "exact", "hash"}
_COST_CLASSES = {"latency", "bandwidth", "compute"}

# Opcodes whose semantics are atomic read-modify-write (R5).
_ATOMIC_OPCODES = ATOMIC_OPCODES
# Control/provenance opcodes realized on the H lane (legal for any plan).
_CONTROL_OPCODES = {
    Opcode.NOP,
    Opcode.PHASE_ENTER,
    Opcode.PHASE_LEAVE,
    Opcode.PROV_NOTE,
    Opcode.BARRIER,
}

# Access patterns whose touched index set is data-dependent (R7): a strict bounds
# contract cannot be discharged statically, so a runtime verify contract is required.
_DATA_DEPENDENT = {StrideClass.CACHELINE, StrideClass.RANDOM}
# The signed 64-bit wire domain every extent, count and cost crosses into (the MLIR
# attributes, the StreamPack fields, the C runtime): an oracle integer that does not fit
# it is not a value the other rails can carry, so the structural laws refuse it here too
# (S0-6: the same bound the op verifiers apply with checked arithmetic).
_I64_MAX = (1 << 63) - 1


def _is_pow2(n: int) -> bool:
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


def verify(module: Module) -> list[Diagnostic]:
    """Module/claim laws R1-R8 (the static half of R8: cost-class completeness)."""
    diags: list[Diagnostic] = []

    # R1: registry uniqueness (RID unique within the module's registry namespace).
    seen: set[int] = set()
    for rid in module.resources:
        if rid in seen:
            diags.append(Diagnostic("R1", f"duplicate RID {rid}"))
        seen.add(rid)

    # R1 (registry well-formedness, S0-6): a registry entry describes a real extent. Every
    # declared shape extent is positive (an empty shape is an UNKNOWN extent, which R7 then
    # cannot check statically -- the MLIR rail's convention), the element count fits the
    # signed 64-bit wire domain and the alignment is a positive power of two. Mirrors the
    # parse-time `bcir.resource` verifier: the oracle used to admit a zero-extent or
    # misaligned resource the law rail refuses at parse. (`elem_bytes` is NOT held here: the
    # dialect's resource carries no element width, and cfront's zero-size objects -- a
    # `void *` pointee, a label address -- legitimately declare 0.)
    for res in module.resources.values():
        for extent in res.shape:
            if extent <= 0:
                diags.append(
                    Diagnostic(
                        "R1", f"resource {res.rid}: shape extents must be positive (got {extent})"
                    )
                )
        if res.shape and all(extent > 0 for extent in res.shape):
            count = 1
            for extent in res.shape:
                count *= extent
            if count > _I64_MAX:
                diags.append(
                    Diagnostic(
                        "R1", f"resource {res.rid}: shape element count exceeds signed 64-bit range"
                    )
                )
        if not _is_pow2(res.align):
            diags.append(
                Diagnostic(
                    "R1",
                    f"resource {res.rid}: align must be a positive power of two (got {res.align})",
                )
            )

    # R1.1: claim-id uniqueness (mirror of R1, for the claim namespace). Every claim id must be unique
    # within the module -- a duplicate/injected claim id makes the claim graph ambiguous (a plan step,
    # an attestation, or the structural digest could silently bind to the wrong claim). The C twin
    # (runtime/c/bcir_verify.c) enforces the same law over the WHOLE unit's claim array; the cfront rail
    # additionally calls cfront_unit_claim_ids_unique() to aggregate across functions (each cfront
    # function is verified as its own single-phase Module, so this per-module pass only catches an
    # intra-function duplicate -- the unit-wide pass catches a cross-function one).
    seen_cid: set[int] = set()
    for ph in module.phases:
        for claim in ph.claims:
            if claim.id in seen_cid:
                diags.append(Diagnostic("R1.1", f"duplicate claim id {claim.id}"))
            seen_cid.add(claim.id)

    # R2: registry resolution -- every claim resource reference resolves.
    for ph in module.phases:
        for claim in ph.claims:
            for rid in claim.io_rids():
                if module.resource(rid) is None:
                    diags.append(
                        Diagnostic("R2", f"claim {claim.id} references undeclared RID {rid}")
                    )

    # R3: domain legality -- claim domain contracts correspond to registry placement.
    for res in module.resources.values():
        if res.access == "ham" and res.domain == Domain.MMIO:
            diags.append(
                Diagnostic("R3", f"resource {res.rid}: HAM access is illegal in the MMIO domain")
            )
    for ph in module.phases:
        for claim in ph.claims:
            touched = [module.resource(rid) for rid in claim.io_rids()]
            touched = [r for r in touched if r is not None]
            if touched and claim.domain not in {r.domain for r in touched}:
                diags.append(
                    Diagnostic(
                        "R3",
                        f"claim {claim.id}: declares domain {claim.domain.name} but touches only "
                        f"{{{', '.join(sorted({r.domain.name for r in touched}))}}}",
                    )
                )
            # R3 (the isolated-domain redirection gap, S0-6 -- one rule on both rails): a
            # resource in a device-ISOLATED domain (MMIO: `model.ISOLATED_DOMAINS`) may be
            # touched only by a claim declaring that domain. A host-domain claim that reaches a
            # device register while "backed" by one RAM operand was accepted by the membership
            # check above -- an isolated resource silently treated as another address space. The
            # converse stays legal: an MMIO claim carries the RAM value it stores / the index it
            # reads (every cfront MMIO access has that shape), so only the RESOURCE side of the
            # pair is required to match.
            for kind, rids in (("read", claim.rd), ("write", claim.wr)):
                for rid in rids:
                    res = module.resource(rid)
                    if (
                        res is not None
                        and res.domain in ISOLATED_DOMAINS
                        and res.domain != claim.domain
                    ):
                        diags.append(
                            Diagnostic(
                                "R3",
                                f"claim {claim.id}: {kind} of RID {rid} (domain {res.domain.name}) "
                                f"does not match the claim domain {claim.domain.name} -- an "
                                f"isolated resource may not be reached as another address space",
                            )
                        )
            for rid in claim.wr:
                res = module.resource(rid)
                if res is not None and res.domain == Domain.MMIO and claim.hazard == "unique":
                    diags.append(
                        Diagnostic(
                            "R3",
                            f"claim {claim.id}: MMIO write to RID {rid} requires an "
                            f"atomic/barriered hazard contract",
                        )
                    )

    # R4 (phase identity, S0-6): a phase id names ONE phase and every dependency names a
    # declared phase. `Module.phase_map()` keys by id, so a duplicate silently shadowed its
    # twin (the shadowed phase's deps vanished from the DAG) and `topological_phase_ids`
    # ignored a dangling dep -- the canonical order was computed over a graph the module did
    # not declare. Both rails now refuse the module before any order is derived from it.
    seen_pid: set[int] = set()
    for ph in module.phases:
        if ph.phase_id in seen_pid:
            diags.append(Diagnostic("R4", f"duplicate phase id {ph.phase_id}"))
        seen_pid.add(ph.phase_id)
    declared_pids = {ph.phase_id for ph in module.phases}
    for ph in module.phases:
        for dep in ph.deps:
            if dep not in declared_pids:
                diags.append(
                    Diagnostic("R4", f"phase {ph.phase_id} depends on undeclared phase {dep}")
                )
    # R4: phase DAG legality (acyclic).
    if _has_cycle(module):
        diags.append(Diagnostic("R4", "phase dependency graph contains a cycle"))

    # R5: hazard legality -- the hazard contract is sufficient for the declared semantics.
    for ph in module.phases:
        for claim in ph.claims:
            if claim.hazard not in _HAZARDS:
                diags.append(
                    Diagnostic("R5", f"claim {claim.id}: unknown hazard contract {claim.hazard!r}")
                )
                continue
            if claim.opcode in _ATOMIC_OPCODES and claim.hazard == "unique":
                diags.append(
                    Diagnostic(
                        "R5",
                        f"claim {claim.id}: atomic opcode {claim.opcode.name} requires an "
                        f"atomic/barriered hazard contract",
                    )
                )
            if claim.lane == Lane.A and claim.hazard == "unique":
                diags.append(
                    Diagnostic(
                        "R5",
                        f"claim {claim.id}: atomic lane A requires an atomic/barriered "
                        f"hazard contract",
                    )
                )
            # §5.14 Phase 2 (indirect-call effect): a dispatch claim's DECLARED callee signature,
            # when carried, must be well-formed "ret(params)" -- a malformed record would poison the
            # R18/commutation consumers silently. Vacuous when absent (the opaque-edge default).
            if claim.callee_sig and "(" not in claim.callee_sig:
                diags.append(
                    Diagnostic(
                        "R18",
                        f"claim {claim.id}: malformed indirect-callee signature "
                        f"{claim.callee_sig!r} (expected 'ret(params)')",
                    )
                )
            # §5.14 Phase 2: a VOLATILE access (MMIO) must carry an ordered hazard -- volatility is
            # an ordering/legality signal, not a cosmetic tag. Vacuous unless a claim opts in.
            if claim.volatile and claim.hazard == "unique":
                diags.append(
                    Diagnostic(
                        "R5",
                        f"claim {claim.id}: volatile access requires an atomic/barriered "
                        f"hazard contract",
                    )
                )
        # CT2 decoupling soundness: the GGG/random tail executes decoupled from the
        # wave order, so a same-phase conflict touching a sparse claim loses its
        # implicit serialization -- both ends must carry an ordered hazard contract.
        # R5 can ONLY fire on a conflict where at least one side is sparse, so precompute
        # sparsity once (not per pair) and skip the whole O(n^2) pair scan when the phase
        # has no sparse claim -- which collapses a large single-phase function (e.g. a 7500-
        # claim body) from ~n^2/2 `_conflict` calls to O(n). Behaviour-identical: the same
        # pairs reach the error condition, in the same order; only the cheap sparse gate now
        # precedes the costly `_conflict` (and the no-sparse phase is provably a no-op here).
        claims = ph.claims
        is_sp = [_is_sparse(c) for c in claims]
        if any(is_sp):
            for i, a in enumerate(claims):
                a_sp = is_sp[i]
                for j in range(i + 1, len(claims)):
                    if not (a_sp or is_sp[j]):
                        continue
                    b = claims[j]
                    if not _conflict(a, b):
                        continue
                    for c in (a, b):
                        if c.hazard == "unique":
                            diags.append(
                                Diagnostic(
                                    "R5",
                                    f"claim {c.id}: conflicts across the decoupled GGG tail in "
                                    f"phase {ph.phase_id} without an atomic/barriered hazard",
                                )
                            )

    # R6: lane legality -- lane type matches the declared access pattern.
    for ph in module.phases:
        for claim in ph.claims:
            legal = _LEGAL_LANES.get(claim.stride_class, set())
            if claim.lane not in legal:
                diags.append(
                    Diagnostic(
                        "R6",
                        f"claim {claim.id}: lane {claim.lane.name} illegal for "
                        f"stride_class {claim.stride_class.name}",
                    )
                )

    # R7: bounds legality -- strict bounds are discharged statically (affine
    # patterns) or guarded by a runtime verify contract (data-dependent patterns).
    for ph in module.phases:
        for claim in ph.claims:
            if claim.bounds not in _BOUNDS:
                diags.append(
                    Diagnostic("R7", f"claim {claim.id}: unknown bounds mode {claim.bounds!r}")
                )
                continue
            if claim.verify not in _VERIFY:
                diags.append(
                    Diagnostic("R7", f"claim {claim.id}: unknown verify contract {claim.verify!r}")
                )
                continue
            # R7 (access-pattern well-formedness, S0-6): the iteration extent is non-negative
            # and the stride positive, whatever the bounds mode -- the parse-time `bcir.claim`
            # verifier's rule. The oracle used to fold a zero or negative stride to 1 (`max(1,
            # stride_k)`) and admit a negative count or offset, so a claim the law rail refuses
            # at parse verified clean here and priced as a unit-stride stream.
            malformed = False
            if claim.count < 0:
                diags.append(
                    Diagnostic(
                        "R7", f"claim {claim.id}: count must be non-negative (got {claim.count})"
                    )
                )
                malformed = True
            if claim.offset < 0:
                diags.append(
                    Diagnostic(
                        "R7", f"claim {claim.id}: offset must be non-negative (got {claim.offset})"
                    )
                )
                malformed = True
            if claim.stride_k < 1:
                diags.append(
                    Diagnostic(
                        "R7", f"claim {claim.id}: stride_k must be positive (got {claim.stride_k})"
                    )
                )
                malformed = True
            if malformed:
                continue
            # A `masked` access (the §5.12 promotion: runtime-bounds-checked, the contract the quarantine
            # handler discharges) must DECLARE that runtime contract -- `verify == "bounds"`. The law now
            # SEES the masked metadata it previously skipped: a masked claim with no bounds verify is a
            # promotion the backend would emit without a guard (a silent loss of the check).
            if claim.bounds == "masked" and claim.verify != "bounds":
                why = (
                    f" (extent provenance: {claim.bounds_provenance})"
                    if claim.bounds_provenance
                    else ""
                )
                diags.append(
                    Diagnostic(
                        "R7",
                        f"claim {claim.id}: masked (runtime-bounds-checked) access must carry a "
                        f"'bounds' verify contract, not {claim.verify!r}{why}",
                    )
                )
            if claim.bounds != "strict":
                continue
            if claim.stride_class in _DATA_DEPENDENT:
                if claim.verify == "none":
                    diags.append(
                        Diagnostic(
                            "R7",
                            f"claim {claim.id}: data-dependent {claim.stride_class.name} access "
                            f"with strict bounds requires a runtime verify contract",
                        )
                    )
                continue
            # Affine pattern: the touched extent is statically known. The stride
            # applies to the streamed read source; writes land unit-stride (a
            # conservative under-approximation -- never a false positive). A
            # reduction (op "reduce.*") accumulates count reads into a single
            # location, so its write extent is one element, not count.
            k = claim.stride_k
            read_extent = claim.offset + (claim.count - 1) * k + 1 if claim.count > 0 else 0
            is_reduction = claim.op.startswith("reduce.")
            write_extent = claim.offset + (1 if is_reduction else claim.count)
            if max(read_extent, write_extent) > _I64_MAX:
                # The wire domain (S0-6): an extent the MLIR attributes and the C runtime
                # cannot carry is refused, not compared -- checked arithmetic on the law rail.
                diags.append(
                    Diagnostic(
                        "R7", f"claim {claim.id}: affine access extent exceeds signed 64-bit range"
                    )
                )
                continue
            for rid, extent, kind in [(r, read_extent, "read") for r in claim.rd] + [
                (w, write_extent, "write") for w in claim.wr
            ]:
                res = module.resource(rid)
                if res is None or not res.shape:
                    continue
                if extent > res.count:
                    diags.append(
                        Diagnostic(
                            "R7",
                            f"claim {claim.id}: {kind} of RID {rid} overruns the resource "
                            f"(extent {extent} > {res.count})",
                        )
                    )

    # R8 (static half): cost completeness -- every claim names a known cost class.
    for ph in module.phases:
        for claim in ph.claims:
            if claim.cost_class not in _COST_CLASSES:
                diags.append(
                    Diagnostic("R8", f"claim {claim.id}: unknown cost class {claim.cost_class!r}")
                )

    # EV1-EV3 (event phases, driver roadmap A1/B1): module laws too, so the canonical
    # verifier carries them. Vacuous over every eventless module; the mask/unmask
    # well-formedness sub-law holds wherever such claims appear. They lived only in
    # `kbcir.events`, and an unarmed event phase reported EV2 there while `verify_all`
    # said the module was lawful.
    from ..kbcir.events import check_event_phases

    for msg in check_event_phases(module):
        law, _, text = msg.partition(":")
        diags.append(Diagnostic(law.strip(), text.strip()))

    return diags


# --- the cross-rail PER-CLAIM STRUCTURAL DIGEST (the count->structural parity fix) ------------------
# The dual-rail parity gate used to compare the two cfront rails by a 9-INTEGER count summary only, so
# any corruption that PRESERVES the counts slipped through: swapping operands between two same-op
# claims, redirecting a call @foo->@bar (both defined), substituting one c.bin.* op for another. This
# digest closes that gap with a CANONICAL, language-independent serialization of every function's claim
# DATAFLOW, hashed with FNV-1a (64-bit). The C twin (bcir_cfront_digest in runtime/c/bcir_cfront.c)
# builds the SAME records and the SAME hash, so the two rails produce a BYTE-IDENTICAL digest, proven
# empirically over the whole fixture corpus (the --canon diff is EMPTY; see
# bcir/tests/test_ir_structural_parity.py).
#
# WHY DATAFLOW VALUE-NUMBERS, not raw positions/rids (empirically forced by the C twin's --canon dump):
# the two cfront frontends are NOT byte-identical IR producers. Three benign divergences would defeat a
# naive "positional rid serialization" and were each measured against the C --canon over all 122
# fixtures:
#   (1) RID base: the Python rid is the C rid + 1, and the absolute values are rail-private.
#   (2) Claim ORDER: 11 fixtures evaluate sibling sub-expressions in a different order (e.g. a*b+c vs
#       c... ), so claim POSITION is not a cross-rail invariant -- a position-sensitive digest diverges.
#   (3) Operand ORDER within a multi-read claim (c.store's base/index/value) differs on a few fixtures.
# A digest that is invariant to (1)-(3) yet still catches the mandated corruptions is the per-function
# multiset of each claim's DATAFLOW VALUE-NUMBER:
#
#   record(claim) = <op-base>|<opcode-int>|<read value-numbers>|<semantic imm>|<dom-int>
#   value_number(rid) = <op-base>(<vns of that producer claim's reads>)  if some claim writes rid;
#                       else "in:p<j>" if rid is the j-th PARAMETER (position is cross-rail stable, so
#                       the two params in `a - b` are distinguished); else the literal "in" (any other
#                       input -- a global / uninitialized local -- stays anonymous, rail-private rid out).
#   plus a per-function OBSERVABLE-OUTPUT anchor:  ret=<vn of the return value>|stores=<dest->value;...>
#   (the LAST-writer value-numbers of what the function RETURNS / STORES -- see _canon_func_records).
#
# - op-base = the `op` string up to the first ':', BUT only for the ops whose ':' suffix is a
#   rail-divergent label the two rails spell differently -- which is ONLY `c.call.vaarg` (the Python
#   oracle emits bare `c.call.vaarg`; the C twin emits `c.call.vaarg:int`). Every OTHER ':' suffix is
#   STRUCTURAL and KEPT: the callee in `c.call:foo` (a redirect @foo->@bar changes it), the WIDTH in
#   `c.cast:uint8_t` (a width change is a real type corruption), and the VALUE in `c.fconst:1.0`.
# - opcode/domain: their INTEGER values (Opcode/Domain are IntEnum valued 0..17 / 0..5 to match the C
#   bcir_opcode/bcir_domain enums by construction -- no enum->name table can drift between rails).
# - SEMANTIC imm is folded per op (see _vn_imm): c.const's value; the struct member BYTE OFFSET (c.load
#   imm[0] / c.addrof / c.store imm[0]); the bitfield BIT-OFFSET/WIDTH/SIGN (c.bf.get/c.bf.set). So
#   tampering a constant (5->999), reading the wrong member (s->x vs s->y), or the wrong/oppositely-
#   signed bitfield (p->a vs p->b) changes the digest. Only the cross-rail-STABLE imm positions are
#   folded; rail-divergent metadata (the c.load bounds `ub`, the c.store trailing _Bool/stride flag) is
#   dropped, preserving byte-identity.
# - the OBSERVABLE-OUTPUT anchor pins what the function actually emits -- the RETURN value's last-writer
#   VN and the sorted STORE (dest-VN -> value-VN) pairs -- so redirecting a SINK claim's write target (a
#   dead/return-temp copy: `return t` -> `return u`; a store destination) changes the digest even though
#   no per-claim record does (wr is rail-private, excluded from the per-claim record). last==first for a
#   single-write rid, so the anchor is byte-identical cross-rail.
# - read-operand order: POSITIONAL BY DEFAULT (so reversing the reads of a non-commutative op -- sub /
#   div / mod / shl / shr / the lt/gt/le/ge comparisons / a c.store's base,index,value -- changes the
#   record, since emit.py lowers `ref(rd[0]) op ref(rd[1])` in order and the reversal is
#   execution-observable). Reads are SORTED ONLY for the genuinely COMMUTATIVE ops (add/mul/and/or/xor/
#   eq/ne), where order cannot affect the value; that sort is what absorbs the few cross-rail operand-
#   order divergences on those ops. The per-function record list is SORTED (absorbs divergence (2): the
#   11 sibling-eval-order fixtures). NOP-marker claims are skipped (matches the count).
#
# This is rid invariant + claim-order invariant (so it matches cross-rail, 171/171) but a real STRUCTURE
# check: an operand swap (commutative OR non-commutative -- the latter now via positional order)
# rebinds value-number trees; an op substitution changes a base; a call redirect changes a `c.call:NAME`
# base; a constant tamper changes c.const's imm; a cast-width change changes `c.cast:WIDTH`; an injected/
# duplicate id is caught by the dedicated unit-wide R1.1 law. Non-tautological in test_ir_structural_parity.py.
_DIGEST_OFFSET = 1469598103934665603  # FNV-1a 64-bit offset basis (== the C offset)
_DIGEST_PRIME = 1099511628211  # FNV-1a 64-bit prime
_DIGEST_MASK = (1 << 64) - 1
_NOP = 0  # Opcode.NOP integer value (the control-marker opcode)
_VN_MAXDEPTH = 96  # recursion guard (== the C twin's; a deep/cyclic chain folds to "cyc")
# The ONLY op whose ':' suffix is a rail-divergent label (Python `c.call.vaarg` vs the C twin
# `c.call.vaarg:int`); its suffix is stripped. Every other ':' suffix is structural and KEPT.
_VN_STRIP_SUFFIX = ("c.call.vaarg",)
# Genuinely COMMUTATIVE ops: operand order cannot change the computed value, so their reads are sorted
# (this is what absorbs the few cross-rail operand-order divergences). All other ops keep POSITIONAL
# read order, so reversing a non-commutative op's operands is caught.
_VN_COMMUTATIVE = frozenset(
    ("c.bin.add", "c.bin.mul", "c.bin.and", "c.bin.or", "c.bin.xor", "c.bin.eq", "c.bin.ne")
)


def _vn_base(op: str) -> str:
    """The op identity for value-numbering: strip ONLY `c.call.vaarg`'s rail-divergent `:T` suffix; keep
    every other ':' suffix (the c.call callee, the c.cast width, the c.fconst value -- all structural)."""
    head = op.split(":", 1)[0]
    return head if head in _VN_STRIP_SUFFIX else op


def _vn_imm(c) -> str:
    """The SEMANTIC immediate component of a claim's record -- the imm fields that encode WHICH datum a
    claim touches (a constant value, a struct member byte offset, a bitfield bit-offset/width/sign), so
    reading/writing the WRONG member or bitfield is caught. Only the cross-rail-STABLE positions are
    folded (the C twin emits the same bytes); rail-divergent metadata (the c.load bounds `ub`, the
    c.store trailing _Bool / stride flags) is dropped, preserving --canon byte-identity:
      c.const           -> all imm (the literal value);
      c.load            -> imm[0], the member BYTE OFFSET (0 if absent; the `ub` at imm[1] is dropped);
      c.store           -> imm[0],imm[1], the (byte offset, unit size) (the _Bool/stride tail dropped);
      c.addrof          -> all imm (&member offset [+ array stride] -- agrees on both rails);
      c.bf.get/c.bf.set -> all imm (bit offset, bit width, signedness -- agrees on both rails);
      c.call.imember    -> all imm (the arrow/dot dispatch flag);
      c.sizeof.vla      -> all imm (the element size).
    A digest-relevant imm change (s->x vs s->y offset, a bitfield p->a vs p->b layout, a signed-vs-
    unsigned bitfield) thus moves the digest, while non-member loads stay cross-rail identical."""
    op = c.op
    imm = [int(v) for v in c.imm]
    if op in ("c.const", "c.addrof", "c.bf.get", "c.bf.set", "c.call.imember", "c.sizeof.vla"):
        keep = imm
    elif op == "c.load":
        keep = [imm[0] if imm else 0]  # the member byte offset (drop the divergent bound)
    elif op == "c.store":
        keep = imm[:2]  # (byte offset, unit size); drop the _Bool/stride tail
    else:
        return ""
    return ",".join(str(v) for v in keep)


def _canon_func_records(lf) -> list[str]:
    """The sorted multiset of per-claim dataflow value-number records for one lowered function, plus an
    OBSERVABLE-OUTPUT anchor line. The per-claim multiset alone is blind to a SINK claim's write target
    (redirecting a dead/return-temp/store wr is invisible if its result is not read downstream); the
    anchor closes that by pinning what the function actually OUTPUTS -- the value it RETURNS and the
    memory it STORES -- as rail-stable value-number trees."""
    claims = [c for c in lf.claims if int(c.opcode) != _NOP]
    first_writer: dict[int, int] = {}  # rid -> index of the FIRST claim that writes it
    last_writer: dict[int, int] = {}  # rid -> index of the LAST claim that writes it
    for i, c in enumerate(claims):
        for w in c.wr:
            first_writer.setdefault(int(w), i)
            last_writer[int(w)] = i
    # A function-INPUT rid (one no claim writes) is a param or a global/uninitialized read. Its absolute
    # rid is rail-private, but a PARAMETER's POSITION is cross-rail stable, so an input that is the j-th
    # parameter value-numbers to "in:pj" -- which DISTINGUISHES the two params in `a - b` vs `b - a` (a
    # non-commutative reversal of two params is now caught). Any other input stays anonymous "in".
    param_ix = {int(rid): j for j, (_nm, rid, _ty) in enumerate(getattr(lf, "params", []))}

    def _ordered(c, parts: list[str]) -> list[str]:
        # commutative ops: sort the reads (order-irrelevant, absorbs cross-rail divergence); else keep
        # POSITIONAL order so reversing a non-commutative op's operands is a real, caught change.
        return sorted(parts) if _vn_base(c.op) in _VN_COMMUTATIVE else parts

    def _vn(writer: dict, memo: dict):
        def vn(rid: int, depth: int) -> str:
            i = writer.get(rid)
            if i is None:
                j = param_ix.get(int(rid))
                return (
                    f"in:p{j}" if j is not None else "in"
                )  # a param (positional, cross-rail) else anon
            if depth > _VN_MAXDEPTH:
                return "cyc"
            if i in memo:
                return memo[i]
            memo[i] = "cyc"  # cycle guard (a loop-carried rid resolves to "cyc")
            c = claims[i]
            parts = _ordered(c, [vn(int(r), depth + 1) for r in c.rd])
            memo[i] = "{}({})".format(_vn_base(c.op), ",".join(parts))
            return memo[i]

        return vn

    # Per-claim records: the FIRST-writer dataflow VN (the form that holds cross-rail byte-identity).
    vn_first = _vn(first_writer, {})
    recs = []
    for c in claims:
        parts = _ordered(c, [vn_first(int(r), 0) for r in c.rd])
        recs.append(
            "{}|{}|{}|{}|{}".format(
                _vn_base(c.op), int(c.opcode), ",".join(parts), _vn_imm(c), int(c.domain)
            )
        )
    recs.sort()

    # The OBSERVABLE-OUTPUT anchor (LAST-writer VN -- a use observes the most-recent prior write, which
    # is what the emitted C returns/stores). The RETURN value's VN catches a sink-wr redirect (e.g.
    # making the `u=a+b` copy write the return temp turns `return t` into `return (a+b)` -- the anchor
    # changes though no per-claim record does). The STORE destinations+values (sorted, rail-stable VNs)
    # catch a dead/store-target redirect. last==first for a single-write rid, so this stays byte-identical.
    vn_last = _vn(last_writer, {})
    rr = getattr(lf, "return_rid", None)  # None for a void function (no returned value)
    ret = vn_last(int(rr), 0) if rr is not None else "void"
    stores = sorted(
        "{}->{}".format(
            vn_last(int(c.rd[0]), 0) if c.rd else "?", vn_last(int(c.rd[-1]), 0) if c.rd else "?"
        )
        for c in claims
        if c.op == "c.store"
    )
    recs.append("ret={}|stores={}".format(ret, ";".join(stores)))
    return recs


def cfront_structural_canon(lowered) -> str:
    """The raw canonical serialization the digest hashes -- the byte-identity proof artifact (the Python
    canon must equal the C twin's `--canon` dump byte-for-byte on the corpus, so the digests match).
    One sorted dataflow value-number record per non-marker claim, with a '@' line per function
    boundary. See the module note above for the exact record format and the empirical justification."""
    out: list[str] = []
    for lf in lowered.functions.values():
        out.extend(_canon_func_records(lf))
        out.append("@")
    return "\n".join(out) + "\n"


def cfront_structural_digest(lowered) -> int:
    """The cross-rail per-claim structural digest of a lowered C unit (the C twin is
    `bcir_cfront_digest`). FNV-1a (64-bit) over `cfront_structural_canon`'s bytes; returns a 64-bit int
    (the dual-rail summary stamps it as `digest=<16-hex>`). Byte-identical to the C twin's digest."""
    h = _DIGEST_OFFSET
    for byte in cfront_structural_canon(lowered).encode("utf-8"):
        h = ((h ^ byte) * _DIGEST_PRIME) & _DIGEST_MASK
    return h


def cfront_unit_claim_ids_unique(lowered) -> list[Diagnostic]:
    """R1.1 across the WHOLE cfront unit: claim ids are unit-wide unique (the cid base is bumped per
    function, so every claim id is globally distinct by construction). The cfront pipeline verifies each
    function as its own single-phase Module, so the per-module R1.1 in verify() only sees an
    intra-function duplicate -- this pass aggregates all functions' claim ids to catch a CROSS-function
    duplicate too, exactly as the C twin's unit-wide bcir_verify_unit R1.1 does."""
    diags: list[Diagnostic] = []
    seen: set[int] = set()
    for lf in lowered.functions.values():
        for c in lf.claims:
            if c.id in seen:
                diags.append(Diagnostic("R1.1", f"duplicate claim id {c.id}"))
            seen.add(c.id)
    return diags


def _same_realization(chosen, offered) -> bool:
    """Two candidates denote the same realization.

    Compared field by field rather than by dataclass equality because `optimize` legally
    rewrites a candidate's *coupled* cost (fusion discounts, thermal derating) while
    leaving the realization identical -- so `base` is compared and the coupled total is
    checked separately by R8.
    """
    return (
        chosen.lane is offered.lane
        and chosen.width == offered.width
        and chosen.name == offered.name
        and chosen.base.v == offered.base.v
    )


def _admissible_candidates(module: Module, h) -> dict[int, tuple]:
    """Re-derive, for each claim, the candidate set the planner may choose from.

    Imported lazily: the verifier is kept free of a hard dependency on the planner so it
    can be read as an independent statement of the laws, and only this one law needs to
    reconstruct what the planner would have offered.
    """
    from ..kbcir.realize import fused_candidates

    # The planner draws from `fused_candidates`, not from the per-claim `candidates_for`:
    # the deforestation and CSE discounts are baked into a consumer's BASE cost there.
    # Re-deriving from `candidates_for` rejected the planner's own plan on every fused
    # consumer -- 3,840 of the 4,096 claims of the audit's matmul fixture -- a false
    # positive that had kept every real caller from passing `h` at all.
    return {cid: tuple(cands) for cid, cands in fused_candidates(module, h).items()}


def verify_plan(
    module: Module, result, h=None, theta=None, policy=None, budget=None
) -> list[Diagnostic]:
    """K_BCIR plan laws R8 (cost completeness) and R9 (plan legality).

    `result` is a `kbcir.realize.RealizationResult` (duck-typed to keep the
    verifier dependency-free).

    `theta` (with `h`) makes R9 re-derive every step's realized cost exactly as the planner
    priced it -- the candidate's base, coupled with the context factor of the step before
    it, under the phase weights of (h, theta, `policy`; PERF when unnamed). A legitimate
    candidate carrying a forged cost keeps the score sum consistent and passes every other
    check; only the re-derivation sees it. `budget` (needs `theta`) makes R(pi, Theta) <= B
    part of legality, as the LangRef central equation states it: an infeasible plan is
    refused with the dimensions named.

    `h` is the target profile the plan was made for. Supply it wherever it is known --
    without it R9 can only ask whether the chosen lane suits the claim's declared
    geometry, and *any* candidate satisfying that passes. A `Candidate(Lane.U, width=3,
    name="forged", cost=0)` for a `vector_add` verified clean: a width the hardware
    cannot issue, a name that denotes no realization, and zero cost. With `h` the law
    re-derives the candidate set from the module and target and requires the chosen
    realization to be a member of it, which is what "the plan is legal" has to mean if a
    certificate is going to rest on it.
    """
    diags: list[Diagnostic] = []
    claims = {c.id: c for ph in module.phases for c in ph.claims}
    admissible = _admissible_candidates(module, h) if h is not None else None

    seen: set[int] = set()
    total = 0
    for step in result.steps:
        claim = claims.get(step.claim_id)
        if claim is None:
            diags.append(Diagnostic("R9", f"plan realizes unknown claim {step.claim_id}"))
            continue
        if step.claim_id in seen:
            diags.append(Diagnostic("R9", f"claim {step.claim_id} realized more than once"))
        seen.add(step.claim_id)

        cand = step.candidate
        # R8: every realized step carries a complete, non-negative scalarized cost.
        if len(cand.base.v) != 12:
            diags.append(
                Diagnostic("R8", f"claim {step.claim_id}: candidate cost vector is not 12-d")
            )
        if step.cost < 0:
            diags.append(
                Diagnostic("R8", f"claim {step.claim_id}: negative realized cost {step.cost}")
            )
        total += step.cost

        # R9: the realization has to be one the planner could actually have generated
        # for this claim on this target -- name, lane, width and base cost together.
        if admissible is not None:
            offered = admissible.get(step.claim_id, ())
            if not any(_same_realization(cand, c) for c in offered):
                diags.append(
                    Diagnostic(
                        "R9",
                        f"claim {step.claim_id}: realization "
                        f"{cand.name!r}({cand.lane.name}, width {cand.width}) is not among "
                        f"the {len(offered)} candidate(s) this target admits: "
                        f"{sorted(c.name for c in offered)}",
                    )
                )

        # R9: an ATOMIC opcode keeps an atomic realization. Checked before the geometry
        # rules because it does not depend on them: `stride_class` describes which
        # elements are touched, and no answer to that question makes a vectorized or
        # gathered read-modify-write atomic. The geometry check alone passed
        # ATOMIC_ADD/SCALAR realized as `U vec16` and ATOMIC_ADD/RANDOM as `GGG gather`.
        if claim.opcode in _ATOMIC_OPCODES and (cand.lane is not Lane.A or cand.width != 1):
            diags.append(
                Diagnostic(
                    "R9",
                    f"claim {step.claim_id}: atomic opcode {claim.opcode.name} realized as "
                    f"{cand.lane.name} width {cand.width}; an atomic read-modify-write has "
                    f"one realization, A lane width 1",
                )
            )

        # R9: the chosen realization is legal for the claim's declared geometry.
        if cand.lane == Lane.H:
            if claim.opcode not in _CONTROL_OPCODES and claim.stride_class != StrideClass.SCALAR:
                diags.append(
                    Diagnostic(
                        "R9",
                        f"claim {step.claim_id}: H-lane realization {cand.name!r} for a "
                        f"non-control claim",
                    )
                )
        elif cand.lane not in _LEGAL_LANES.get(claim.stride_class, set()):
            diags.append(
                Diagnostic(
                    "R9",
                    f"claim {step.claim_id}: chosen lane {cand.lane.name} illegal for "
                    f"stride_class {claim.stride_class.name}",
                )
            )

    # R9 (scope): every realized cost re-derives from (h, theta, policy) -- the same
    # predicate the planner prices its DAG edges with (`realize.edge_cost`).
    if theta is not None and h is not None:
        from ..kbcir.realize import step_cost
        from ..kbcir.weights import PERF

        pol = policy if policy is not None else PERF
        prev = None
        for step in result.steps:
            if step.claim_id not in claims:
                continue
            expected = step_cost(prev, step.candidate, h, theta, step.phase_id, pol)
            if expected != step.cost:
                diags.append(
                    Diagnostic(
                        "R9",
                        f"claim {step.claim_id}: realized cost {step.cost} does not re-derive "
                        f"from the scope (expected {expected} for {step.candidate.name!r} "
                        f"under policy {pol.name!r})",
                    )
                )
            prev = step.candidate
    if budget is not None:
        if theta is None:
            raise ValueError(
                "R9 budget feasibility needs theta: R(pi, Theta) is priced under Theta"
            )
        from ..kbcir.cost import DIMS
        from ..kbcir.rcsp import plan_resources

        used = plan_resources(result, theta)
        over = [f"{DIMS[d]} {used.v[d]} > {cap}" for d, cap in budget.caps if used.v[d] > cap]
        if over:
            diags.append(Diagnostic("R9", "plan exceeds its budget: " + ", ".join(over)))

    # R9: total coverage -- a plan must realize every claim exactly once.
    for cid in claims:
        if cid not in seen:
            diags.append(Diagnostic("R9", f"plan does not realize claim {cid}"))

    # R9: the reported score is the sum of the realized step costs.
    if result.steps and total != result.score:
        diags.append(Diagnostic("R9", f"plan score {result.score} != sum of step costs {total}"))

    # R9: steps follow the topological phase order.
    pos = {pid: i for i, pid in enumerate(_topo_phase_ids(module))}
    last = -1
    for step in result.steps:
        p = pos.get(step.phase_id, -1)
        if p < last:
            diags.append(Diagnostic("R9", f"claim {step.claim_id}: realized out of phase order"))
            break
        last = max(last, p)

    return diags


def verify_pack(module: Module, pack, result=None) -> list[Diagnostic]:
    """GEM stream laws R10 (provenance) and R11 (generation validity).

    `pack` is a `gem.streampack.StreamPack` (duck-typed).

    `result` is the plan the pack was hydrated from. Supply it wherever it is known.
    Without it R10 can only ask whether each segment maps back to *some* claim in the
    module -- it cannot ask whether the pack is the lowering of the plan that was priced
    and certified. `verify_all` checked plan and pack independently and never proved the
    one came from the other, so a pack hydrated from a `vec4` plan verified clean against
    a `scalar` plan: the artifact chain graph -> plan -> pack had no link.
    """
    diags: list[Diagnostic] = []
    claims = {c.id for ph in module.phases for c in ph.claims}

    # R10: coverage. Every claim the module declares needs a segment, or the pack does
    # not realize the module. An EMPTY pack made every loop below vacuous and verified
    # clean -- the whole provenance law held trivially over nothing.
    covered = {s.claim_id for s in pack.segments}
    for cid in sorted(claims - covered):
        claim = next(c for ph in module.phases for c in ph.claims if c.id == cid)
        if claim.opcode in _CONTROL_OPCODES:
            continue  # control claims lower to no stream segment
        diags.append(
            Diagnostic(
                "R10",
                f"claim {cid} has no StreamPack segment; the pack does not realize the module",
            )
        )
    segment_ids = [s.claim_id for s in pack.segments]
    trace_ids = [t.claim_id for t in pack.trace_notes]
    prefetch_names = [p.name for p in pack.prefetches]
    traced = {t.claim_id for t in pack.trace_notes}
    prefetches = {p.name for p in pack.prefetches}
    pf_targets = {p.name: set(p.targets) for p in pack.prefetches}

    # R10: stream structure -- v2 pipeline/double-buffer contracts are well-formed.
    if len(set(segment_ids)) != len(segment_ids):
        diags.append(Diagnostic("R10", "duplicate segment claim_id in StreamPack"))
    if len(set(trace_ids)) != len(trace_ids):
        diags.append(Diagnostic("R10", "duplicate trace claim_id in StreamPack"))
    if len(set(prefetch_names)) != len(prefetch_names):
        diags.append(Diagnostic("R10", "duplicate prefetch name in StreamPack"))
    if getattr(pack, "pipeline_depth", 1) < 1:
        diags.append(
            Diagnostic("R10", f"invalid pipeline_depth {pack.pipeline_depth} (must be >= 1)")
        )
    for pf in pack.prefetches:
        if getattr(pf, "buffers", 1) not in (1, 2):
            diags.append(
                Diagnostic("R10", f"prefetch {pf.name}: invalid buffer count {pf.buffers} (1 or 2)")
            )

    # R10: stream provenance -- every segment maps back to a live BCIR claim.
    for seg in pack.segments:
        if seg.claim_id not in traced:
            diags.append(
                Diagnostic("R10", f"segment {seg.name}: no trace note for claim {seg.claim_id}")
            )
        if seg.claim_id not in claims:
            diags.append(
                Diagnostic("R10", f"segment {seg.name}: references unknown claim {seg.claim_id}")
            )
        for rid in tuple(seg.reads) + tuple(seg.writes):
            if module.resource(rid) is None:
                diags.append(
                    Diagnostic("R10", f"segment {seg.name}: references undeclared RID {rid}")
                )
        if seg.prefetch is not None and seg.prefetch not in prefetches:
            diags.append(
                Diagnostic("R10", f"segment {seg.name}: undeclared prefetch {seg.prefetch!r}")
            )
        # R10: a declared prefetch must actually FEED this segment -- at least one of its
        # read RIDs must be a prefetch target (hydrate sets pf.targets == claim.rd). A
        # redirected/swapped target (no read covered) is a broken provenance binding the
        # freestanding C twin (bcir_sp_verify_semantic) also rejects, so the rails agree.
        elif seg.prefetch is not None and seg.reads:
            tgts = pf_targets.get(seg.prefetch, set())
            if not (set(seg.reads) & tgts):
                diags.append(
                    Diagnostic(
                        "R10",
                        f"segment {seg.name}: prefetch {seg.prefetch!r} feeds no read RID "
                        f"(targets {sorted(tgts)} disjoint from reads {sorted(seg.reads)})",
                    )
                )

    # R10: the pack is the lowering of THIS plan. Segment-to-claim provenance is not
    # enough on its own: the claim ids can all be right while the realization the pack
    # encodes is one the plan never chose, and the certificate prices the plan.
    if result is not None:
        chosen = {step.claim_id: step.candidate for step in result.steps}
        for seg in pack.segments:
            cand = chosen.get(seg.claim_id)
            if cand is None:
                diags.append(
                    Diagnostic(
                        "R10",
                        f"segment {seg.name}: claim {seg.claim_id} is not realized by "
                        f"the plan this pack is verified against",
                    )
                )
                continue
            width = getattr(seg, "width", None)
            if width is not None and int(width) != int(cand.width):
                diags.append(
                    Diagnostic(
                        "R10",
                        f"segment {seg.name}: width {width} but the plan chose "
                        f"{cand.name!r} at width {cand.width}",
                    )
                )
            lane = getattr(seg, "lane", None)
            if lane is not None and int(lane) != int(cand.lane):
                diags.append(
                    Diagnostic(
                        "R10",
                        f"segment {seg.name}: lane {lane} but the plan chose "
                        f"{cand.lane.name} ({int(cand.lane)})",
                    )
                )

    # R11: generation validity -- the pack's tags match the live registry. A
    # mismatch is a stale pack: rehydrate (keep/patch/repack/replan,
    # kbcir.calibrate.rehydrate_decide), never execute silently.
    if pack.topo_gen < 1:
        diags.append(Diagnostic("R11", f"invalid topo_gen {pack.topo_gen} (must be >= 1)"))
    reg_map = max((r.map_gen for r in module.resources.values()), default=0)
    reg_data = max((r.data_gen for r in module.resources.values()), default=0)
    if pack.map_gen != reg_map:
        diags.append(
            Diagnostic(
                "R11",
                f"stale StreamPack: map_gen {pack.map_gen} != registry {reg_map} "
                f"(rehydrate: repack)",
            )
        )
    if pack.data_gen != reg_data:
        diags.append(
            Diagnostic(
                "R11",
                f"stale StreamPack: data_gen {pack.data_gen} != registry {reg_data} "
                f"(rehydrate: replan)",
            )
        )

    return diags


# The textual instruction surface emit_kernel_ll may legally produce (R12: no
# invented opcodes; the emitter is legal-IR-only).
_LEGAL_RESULT_OPS = {
    "phi",
    "getelementptr",
    "load",
    "icmp",
    "add",
    "sub",
    "mul",
    "fadd",
    "fsub",
    "fmul",
}
_LEGAL_STMT_OPS = {"store", "br", "ret", "fence"}
_RESULT_RE = re.compile(r"^%[\w.]+\s*=\s*(\w+)")
_GUARD_RE = re.compile(r"icmp\s+\w+\s+i64\s+%\w+,\s*%n\b")
_WIDTH_RE = re.compile(r"\bwidth=(\d+)\b")


def verify_support_preservation(source, target, mapping=None) -> list[Diagnostic]:
    """Objective-support law R12 (refinement): a mapping function must preserve
    where the objective matters -- `f(Supp(J)) ⊆ Supp(J')`. A lowering may
    sharpen, rescale, or fuse a cost dimension, but it may not silently *drop* one
    that was nonzero (lose the thermal/security/accuracy/verification term) unless
    `mapping` explicitly discharges it (the same escape R12 grants bounds/hazard/
    precision). The objective's footprint is an invariant of legal lowering.

    `source`/`target` are `kbcir.cost.CostVector`s (J, J'); `mapping` a
    `kbcir.mapping.MappingFunction` (defaults to the identity dimension map with
    no discharges).
    """
    from ..kbcir.mapping import MappingFunction

    f = mapping or MappingFunction(name="identity")
    diags: list[Diagnostic] = []
    for d in sorted(f.dropped(source, target)):
        tgt = f.dim_map.get(d, d)
        diags.append(
            Diagnostic(
                "R12",
                f"objective support not preserved: source dimension {d!r} is nonzero "
                f"but maps to {tgt!r} which the target drops, with no discharge "
                f"(map {f.name!r})",
            )
        )
    return diags


def verify_commutativity(square, inputs, eq=None) -> list[Diagnostic]:
    """Commutativity / path-independence law R12 (parity): two conversion paths
    that reach the same target must agree -- `Λ ∘ Ψ = Φ`. The PARITY/manifest
    discipline generalized to any representation rail: a result may not depend on
    which legal path produced it. `square` is a `kbcir.mapping.CommutingSquare`.
    """
    diags: list[Diagnostic] = []
    for x in square.mismatches(inputs, eq):
        diags.append(
            Diagnostic(
                "R12",
                f"conversion paths disagree on {x!r}: lam(psi(x)) != phi(x) -- the "
                f"square {square.name!r} does not commute (Λ∘Ψ ≠ Φ)",
            )
        )
    return diags


def verify_lowering(
    module: Module, result, ll_text: str, elem: str = "f32", width_override: int | None = None
) -> list[Diagnostic]:
    """Lowering law R12: the emitted LLVM IR preserves the BCIR semantic (lane
    geometry, bounds, hazard, precision) or carries an explicit discharge note.

    Checks the textual kernel produced by `lower.llvm.emit_kernel_ll` against the
    K_BCIR-selected realization. The head comment is the discharge record: it must
    declare the realized width, and that width must be the selected one or the
    documented scalar legalization (count not divisible by the vector width).
    """
    from ..lower.llvm import _FOP, _IOP, find_elementwise
    from ..lower.memory_model import hazard_to_ordering

    diags: list[Diagnostic] = []
    try:
        claim, cand = find_elementwise(module, result)
    except NotImplementedError:
        return [Diagnostic("R12", "no lowerable elementwise claim selected in this plan")]

    # Discharge note: the head comment carries the realized lane geometry.
    head = ll_text.splitlines()[0] if ll_text else ""
    if not head.startswith("; BCIR -> LLVM IR"):
        return [Diagnostic("R12", "missing lowering discharge note (head comment)")]
    m = _WIDTH_RE.search(head)
    if m is None:
        return [Diagnostic("R12", "discharge note does not declare a width")]
    declared_w = int(m.group(1))

    # Lane geometry: the realized width is the K_BCIR-selected width, or the
    # documented scalar legalization when the trip count is not divisible.
    n = max(1, claim.count)
    base_w = width_override if width_override else cand.width
    expected_w = base_w if (base_w >= 1 and n % base_w == 0) else 1
    if declared_w != expected_w:
        diags.append(
            Diagnostic(
                "R12",
                f"lane geometry not preserved: realized width {declared_w} != "
                f"selected width {expected_w} (candidate {cand.name})",
            )
        )

    # Precision + geometry of the kernel op: the compute instruction operates on
    # the contracted element type at the realized width.
    ety = "i32" if elem == "i32" else "float"
    op_ll = _IOP[claim.opcode] if elem == "i32" else _FOP[claim.opcode][0]
    kernel_ty = f"<{declared_w} x {ety}>" if declared_w > 1 else ety
    if f"{op_ll} {kernel_ty}" not in ll_text:
        diags.append(
            Diagnostic(
                "R12", f"precision not preserved: kernel op '{op_ll} {kernel_ty}' not emitted"
            )
        )
    if declared_w == 1 and f"x {ety}>" in ll_text:
        diags.append(
            Diagnostic("R12", "lane geometry not preserved: vector types in a scalar lowering")
        )

    # No invented instructions: every emitted instruction is in the legal set.
    for raw in ll_text.splitlines():
        s = raw.strip()
        if (
            not s
            or s.startswith(";")
            or s.startswith("source_filename")
            or s.startswith("define")
            or s == "}"
            or s.endswith(":")
        ):
            continue
        rm = _RESULT_RE.match(s)
        op = rm.group(1) if rm else s.split()[0]
        if op not in (_LEGAL_RESULT_OPS | _LEGAL_STMT_OPS):
            diags.append(Diagnostic("R12", f"instruction {op!r} outside the legal lowering set"))

    # Bounds: a strict bounds contract is discharged by the trip-count guard.
    if claim.bounds == "strict" and not _GUARD_RE.search(ll_text):
        diags.append(
            Diagnostic("R12", "strict bounds contract not discharged (no trip-count guard on %n)")
        )

    # Hazard: an ordered hazard contract must materialize as a fence.
    ordering = hazard_to_ordering(claim.hazard)
    if ordering in ("acq_rel", "seq_cst") and "fence" not in ll_text:
        diags.append(
            Diagnostic(
                "R12",
                f"hazard contract {claim.hazard!r} requires a fence (>= {ordering}) "
                f"in the lowered kernel",
            )
        )

    return diags


def verify_c_lowering(
    module: Module,
    result,
    c_text: str,
    elem: str = "f32",
    width_override: int | None = None,
    hw_width: int | None = None,
) -> list[Diagnostic]:
    """Lowering law R12 for the portable C backend: the emitted C23 kernel
    (`lower.c_kernel.emit_kernel_c`) preserves the K_BCIR-selected realization --
    lane geometry, precision (the contracted element type), bounds (a trip-count
    guard so the kernel is safe for any n), the elementwise op, and the
    non-aliasing contract (`restrict` when the read and write resources are
    disjoint).

    Lane geometry is *width-aware* (`hw_width` = the target's widest lane,
    `HProfile.vector_width`): the lowering must **honor** the selected width, which
    means different things at the full lane vs a throttle. (a) `w == 1`: a scalar
    loop, no vector cap. (b) `w == hw_width` (go-fast): the idiomatic loop is
    correct -- the compiler realizes >= the lane -- so the kernel must NOT cap below
    the lane (a hidden sub-width cap would throttle the backend). (c) `1 < w <
    hw_width` (a deliberate thermal/power throttle): the kernel MUST cap at `w`
    (a fixed-trip width-`w` loop) so the compiler cannot widen past the sub-maximal
    lane. With no `hw_width` the check conservatively requires the cap for `w > 1`
    (the literal geometry-encoding form -- unchanged from before)."""
    from ..gem import hydrate
    from ..lower.c_kernel import C_OP, _ctype
    from ..lower.llvm import find_elementwise

    diags: list[Diagnostic] = []
    try:
        claim, cand = find_elementwise(module, result)
    except NotImplementedError:
        return [Diagnostic("R12", "no lowerable elementwise claim selected in this plan")]

    head = c_text.splitlines()[0] if c_text else ""
    if not head.startswith("/* BCIR -> portable C"):
        return [Diagnostic("R12", "missing C lowering discharge note (head comment)")]
    m = re.search(r"\bwidth=(\d+)\b", head)
    if m is None:
        return [Diagnostic("R12", "C discharge note does not declare a width")]
    declared_w = int(m.group(1))

    w = int(width_override) if width_override else int(cand.width)
    w = w if w >= 1 else 1
    if declared_w != w:
        diags.append(
            Diagnostic(
                "R12",
                f"lane geometry not preserved: declared width {declared_w} != selected "
                f"width {w} (candidate {cand.name})",
            )
        )

    # Lane geometry in the body, width-aware (see the docstring).
    full_lane = hw_width is not None and w == int(hw_width)
    if w == 1:
        if re.search(r"<\s*\d+u", c_text):
            diags.append(
                Diagnostic(
                    "R12", "lane geometry not preserved: a vector-width loop in a scalar kernel"
                )
            )
    elif full_lane:
        # Go-fast: the idiomatic loop realizes >= the full lane. Geometry is the
        # recorded width (checked above) + a bounds-safe loop; a cap *below* the
        # lane would secretly throttle the backend, so reject it.
        sub_caps = [int(x) for x in re.findall(r"<\s*(\d+)u", c_text) if int(x) < w]
        if sub_caps:
            diags.append(
                Diagnostic(
                    "R12",
                    f"lane geometry not preserved: full-width kernel caps below the "
                    f"hardware lane (found width-{min(sub_caps)} cap, lane is {w})",
                )
            )
    else:
        # Deliberate sub-maximal throttle: the cap MUST be physically honored, else
        # the compiler widens past the selected (thermal/power-limited) lane.
        if f"< {w}u" not in c_text:
            diags.append(
                Diagnostic("R12", f"lane geometry not preserved: no width-{w} throttle cap emitted")
            )

    # Precision: the contracted element type on the kernel signature.
    ctype = _ctype(elem)
    if f"const {ctype} " not in c_text:
        diags.append(
            Diagnostic("R12", f"precision not preserved: kernel does not operate on {ctype}")
        )

    # The elementwise op.
    op = C_OP.get(claim.opcode)
    if op and f"] {op} B" not in c_text and f"] {op} B[i]" not in c_text:
        diags.append(Diagnostic("R12", f"op not preserved: elementwise '{op}' not emitted"))

    # Bounds: a trip-count guard on n (the strict bounds contract -> a loop over n).
    if "< n" not in c_text and "<= n" not in c_text:
        diags.append(Diagnostic("R12", "bounds not preserved: no trip-count guard on n"))

    # Non-aliasing: restrict when the read/write resources are disjoint.
    pack = hydrate(module, result)
    seg = next((s for s in pack.segments if s.claim_id == claim.id), None)
    reads = tuple(seg.reads) if seg else tuple(claim.rd)
    writes = tuple(seg.writes) if seg else tuple(claim.wr)
    if not (set(reads) & set(writes)) and "restrict" not in c_text:
        diags.append(
            Diagnostic(
                "R12",
                "aliasing contract not preserved: disjoint operands but no "
                "restrict-qualified pointers",
            )
        )

    return diags


def verify_address_width(
    triple: str, addr_bits: int, what: str = "device-register address"
) -> list[Diagnostic]:
    """Lowering law R12 (address width, S0-6): a first-class integer address -- the operand of
    an MMIO `volatile_load`/`volatile_store` or an `atomic_rmw`/`atomic_cas` -- is `inttoptr`'d
    to the target's pointer, so its width must EQUAL the target's pointer width. A narrower
    address is zero-extended (an i32 on x86_64 reaches only the low 4 GiB and every address
    above it wraps to a different register); a wider one is truncated. The op-level floor
    (>= 32 bits) still applies without a target; with one, this law is the contract. The
    triple -> width table is `kbcir.cost.pointer_width`, mirrored by `BCIRPassSupport.h`
    `pointerWidthOfTriple` (one table, both rails; the corpus checks every triple in use).
    Vacuous for a triple the table does not know -- except for the floor: an operand narrower
    than `ADDRESS_FLOOR_BITS` is refused whatever the target, as the law rail's op verifiers
    refuse it (the one address rule that holds without a target in scope)."""
    from ..kbcir.cost import ADDRESS_FLOOR_BITS, pointer_width

    if addr_bits < ADDRESS_FLOOR_BITS:
        return [
            Diagnostic(
                "R12",
                f"the {what} is {addr_bits} bits; an address operand is at least "
                f"{ADDRESS_FLOOR_BITS} bits (the inttoptr lowering's floor), whatever the target",
            )
        ]
    width = pointer_width(triple)
    if width is None or addr_bits == width:
        return []
    how = "zero-extended" if addr_bits < width else "truncated"
    return [
        Diagnostic(
            "R12",
            f"the {what} is {addr_bits} bits but the target '{triple}' addresses {width}-bit "
            f"pointers; the inttoptr lowering would leave it {how}",
        )
    ]


def verify_manifest(manifest, module, h, theta, policy=None, artifacts=()) -> list[Diagnostic]:
    """Provenance-manifest law R13: a deployed plan's manifest must (a) hash to the
    inputs/artifacts it claims (tamper-evidence) and (b) reproduce the recorded
    optimal score and plan shape (determinism). A failure means the plan is not
    reproducible from its stated provenance -- the debugging/determinism violation.
    """
    from ..kbcir.provenance import build_manifest
    from ..kbcir.weights import PERF

    diags: list[Diagnostic] = []
    fresh = build_manifest(module, h, theta, policy or PERF, artifacts)
    if fresh.digest != manifest.digest:
        diags.append(
            Diagnostic(
                "R13",
                f"manifest digest {manifest.digest} != recomputed {fresh.digest} "
                f"(changed components: {fresh.diff(manifest)})",
            )
        )
        return diags
    if fresh.score != manifest.score:
        diags.append(
            Diagnostic(
                "R13",
                f"manifest is not reproducible: recorded score {manifest.score} != "
                f"replayed {fresh.score}",
            )
        )
    if fresh.widths != manifest.widths:
        diags.append(Diagnostic("R13", "manifest is not reproducible: replayed plan shape differs"))
    return diags


def verify_memory(mm, generation=None, *, recheck=True) -> list[Diagnostic]:
    """Memory-module fixpoint law R13: an artifact may be frozen and
    generation-tagged as a *memory module* only when it is the fixpoint of
    resolution -- `a = Lim(Res(U))`, `saturated == True` -- never a budget cutoff
    `Res^k(U)`. Freezing a non-saturated extraction pins a non-canonical,
    still-improvable representative as memory and breaks the determinism the
    provenance manifest depends on (two runs that cut off at different `k` need
    not agree). The saturation witness IS the admitting certificate for entry
    into a generation.

    With `recheck` (default), the verifier does not merely trust the recorded
    `saturated` flag: it independently re-resolves the stored representative
    (`Res(Lim(Res(U)))`) and confirms it is a genuine, stable fixpoint -- the
    tamper-evidence analog of `verify_manifest` recomputing the digest.
    """
    from ..kbcir.memory import try_freeze, _costs_for

    diags: list[Diagnostic] = []

    # R13: the fixpoint witness. A budget cutoff is not Lim(Res(U)).
    if not mm.saturated:
        diags.append(
            Diagnostic(
                "R13",
                f"memory module is not a fixpoint: saturated == False after "
                f"{mm.iterations} round(s) -- a budget cutoff Res^k(U), not "
                f"Lim(Res(U)); not admissible as memory",
            )
        )

    # R13: a frozen artifact must be generation-tagged (immutable within a gen).
    if mm.generation < 1:
        diags.append(
            Diagnostic(
                "R13",
                f"memory module has no generation tag (generation {mm.generation}); "
                f"a frozen artifact is immutable within a generation and must carry one",
            )
        )
    elif generation is not None and mm.generation != generation:
        diags.append(
            Diagnostic(
                "R13",
                f"memory module generation {mm.generation} != expected {generation}",
            )
        )

    # R13: independent fixpoint recheck (tamper-evidence). Re-resolving the stored
    # representative must reproduce it, saturated -- idempotence Res(Lim) = Lim.
    if recheck and mm.saturated:
        again = try_freeze(mm.expr, generation=mm.generation, costs=_costs_for(mm.expr))
        if not again.saturated:
            diags.append(
                Diagnostic(
                    "R13",
                    "memory module fails the fixpoint recheck: its representative does "
                    "not re-resolve to saturation (the recorded witness is unsound)",
                )
            )
        elif again.expr != mm.expr:
            diags.append(
                Diagnostic(
                    "R13",
                    "memory module is not idempotent: re-resolving its representative "
                    "yields a different form, so it is not Lim(Res(U)) (tampered or "
                    "frozen before saturation)",
                )
            )
        elif again.fingerprint != mm.fingerprint:
            diags.append(
                Diagnostic(
                    "R13",
                    f"memory module fingerprint {mm.fingerprint} != recomputed "
                    f"{again.fingerprint} (content does not match its hash)",
                )
            )

    return diags


def verify_calibration(cert) -> list[Diagnostic]:
    """Calibration-closure law R13: a closed calibration loop is admissible only
    when its frozen table is generation-tagged (`cal_gen >= 1`) and recalibrating
    never regresses -- the win (the measured cost of *not* recalibrating) is
    `>= 0`. A negative win means the "recalibrated" plan is not the optimum under
    the measured model (a broken loop, a stale rescore, or a tampered
    certificate); a `cal_gen < 1` table is an untagged measurement that may not be
    deployed. `cert` is a `kbcir.calibloop.CalibrationCertificate` (duck-typed).
    """
    diags: list[Diagnostic] = []
    if cert.cal_gen < 1:
        diags.append(
            Diagnostic(
                "R13",
                f"calibration table is not generation-tagged (cal_gen {cert.cal_gen}); "
                f"a measured table must be frozen + tagged before it is deployed",
            )
        )
    if cert.win < 0:
        diags.append(
            Diagnostic(
                "R13",
                f"calibration loop regressed: win {cert.win} < 0 (stale cost "
                f"{cert.stale_cost} < recalibrated {cert.calibrated_cost}); the "
                f"recalibrated plan is not optimal under the measured model",
            )
        )
    return diags


# --- R14-R16: the smart-lowering laws (dual-rail with -bcir-lower-to-llvm) --------
#
# These three already exist as MLIR laws + oracle *gates* (gem.cim / gem.dvfs /
# kbcir.allocator); the functions below add them to the Python verifier so the laws
# are symmetric on both rails (docs/PARITY.md, R14-R16). Each mirrors exactly the
# structural check `mlir/lib/BCIRPasses.cpp` (LowerToLLVMPass) performs.

# R16 on-chip capacity caps -- the static SRAM budgets the MLIR law uses
# (BCIRPasses.cpp): a placement into L1 must fit 64 KiB, into L2 4 MiB. L3/DRAM/HBM
# carry no static cap.
_L1_CAP_BYTES = 64 * 1024
_L2_CAP_BYTES = 4 * 1024 * 1024
# R15 legal DVFS clock range (Q8; 256 == nominal x1.0): 0.25x .. 2x.
_CLOCK_MIN_Q8 = 64
_CLOCK_MAX_Q8 = 512
_CLOCK_NOMINAL_Q8 = 256


def verify_cim(pack) -> list[Diagnostic]:
    """R14 (CIM/PIM dispatch legality): a StreamPack segment dispatched to
    processing-in-memory must be a reduction -- PIM does element-local reduce work,
    not general SIMD. Mirrors the MLIR `-bcir-lower-to-llvm` R14 law
    (`dispatch="pim"` legal only on a `reduce.*` op) and the `gem.cim` gate.
    `pack` is a `gem.streampack.StreamPack` (duck-typed)."""
    diags: list[Diagnostic] = []
    for seg in pack.segments:
        if getattr(seg, "dispatch", "core") == "pim" and not seg.opcode.startswith("reduce."):
            diags.append(
                Diagnostic(
                    "R14",
                    f"segment {seg.name}: pim dispatch illegal for non-reduction op "
                    f"{seg.opcode!r} (claim {seg.claim_id})",
                )
            )
    return diags


def verify_dvfs(plan) -> list[Diagnostic]:
    """R15 (DVFS clock legality): every per-phase clock is a legal Q8 step in
    [64, 512] (0.25x..2x), and a memory-bound phase must not overclock -- more core
    frequency cannot speed bandwidth-bound work. Mirrors the MLIR
    `-bcir-lower-to-llvm` R15 law (clock_q8 in [64,512]; a pim/memory-bound segment
    must not overclock) and the `gem.dvfs` gate. `plan` is a `gem.dvfs.DVFSPlan`
    (duck-typed: `.decisions` with `phase_id`, `clock_q8`, `klass`)."""
    diags: list[Diagnostic] = []
    for d in plan.decisions:
        if not (_CLOCK_MIN_Q8 <= d.clock_q8 <= _CLOCK_MAX_Q8):
            diags.append(
                Diagnostic(
                    "R15",
                    f"phase {d.phase_id}: clock_q8 {d.clock_q8} out of legal range "
                    f"[{_CLOCK_MIN_Q8}, {_CLOCK_MAX_Q8}]",
                )
            )
        elif d.klass == "memory" and d.clock_q8 > _CLOCK_NOMINAL_Q8:
            diags.append(
                Diagnostic(
                    "R15",
                    f"phase {d.phase_id}: memory-bound phase must not overclock "
                    f"(clock_q8 {d.clock_q8} > {_CLOCK_NOMINAL_Q8})",
                )
            )
    return diags


def verify_allocator(module: Module, placement) -> list[Diagnostic]:
    """R16 (allocator placement legality): an on-chip placement must fit -- a
    resource placed in L1 must be <= 64 KiB, in L2 <= 4 MiB (static size =
    count * elem_bytes). The planner may not promote a tensor into an SRAM tier it
    cannot fit. Mirrors the MLIR `-bcir-lower-to-llvm` R16 law and the
    `kbcir.allocator` capacity gate. `placement` is a `kbcir.allocator.Placement`
    (duck-typed: `.tiers` maps rid -> MemTier); L3/DRAM/HBM carry no cap."""
    from ..kbcir.cost import MemTier

    caps = {MemTier.L1: _L1_CAP_BYTES, MemTier.L2: _L2_CAP_BYTES}
    diags: list[Diagnostic] = []
    for rid, tier in placement.tiers.items():
        cap = caps.get(tier)
        if cap is None:
            continue
        res = module.resource(rid)
        if res is None or not res.shape:
            continue  # dynamic/unknown extent: not statically checkable
        # Mirror the MLIR law EXACTLY: BCIRPasses.cpp computes product(shape) * 4
        # (the ResourceOp carries no elem_bytes attr, so the law assumes a 4-byte
        # element). Using 4 here keeps the two rails in lock-step at the cap boundary.
        nbytes = res.count * 4
        if nbytes > cap:
            diags.append(
                Diagnostic(
                    "R16",
                    f"placement {tier.name} does not fit RID {rid} ({nbytes} B > {cap} B)",
                )
            )
    return diags


def verify_accuracy(module: Module) -> list[Diagnostic]:
    """R17 (accuracy-contract legality): a claim that declares an accuracy tolerance
    (`tolerance_ulp > 0`) must realize within it -- its static worst-case Q8-ULP error
    bound (`precision.accuracy_bound`, compensated iff `precision == "compensated"`) must
    not exceed the declared tolerance. A `reduce.*` over `count` terms drifts up to `count`
    ULP with the naive accumulator but only 1 ULP compensated, so a tight tolerance is the
    law that FORCES the compensated realization (`precision="compensated"`, lowered by
    `lower.c_kernel.emit_compensated_reduce_c`). Dual-rail with the MLIR `-bcir-verify` R17
    law. Claims with no declared tolerance (the default) are unconstrained -- a no-op."""
    from ..kbcir.precision import accuracy_bound

    diags: list[Diagnostic] = []
    for ph in module.phases:
        for claim in ph.claims:
            tol = getattr(claim, "tolerance_ulp", 0)
            if tol <= 0:
                continue
            compensated = getattr(claim, "precision", "") == "compensated"
            bound = accuracy_bound(claim, compensated=compensated)
            if bound > tol:
                diags.append(
                    Diagnostic(
                        "R17",
                        f"claim {claim.id} accuracy bound {bound} ULP exceeds tolerance {tol} "
                        f"ULP (precision={claim.precision or 'naive'!r}; a compensated "
                        f"reduction would bound it at 1)",
                    )
                )
    return diags


_SYNC_TYPES = {"", "synchronous", "asynchronous", "mixed"}


def verify_timing(module: Module) -> list[Diagnostic]:
    """R19 (synchronous-timing legality) + R20 (clock-domain-crossing) over the OPTIONAL `claim.timing`
    metadata -- the RTL / synchronous-timing track (§5.11). Claims that carry NO timing (the default,
    `claim.timing is None`) are entirely unconstrained, so the whole scalar / C-frontend subset verifies
    identically with these laws present (the non-disturbance invariant -- the additive seam, exactly like
    R14/R15/R16 are vacuous for the scalar subset and R17 is vacuous without a declared tolerance).

    R19: a declared timing block is internally consistent -- a valid `sync_type`; non-negative
    `latency_cycles` / `setup_hold_margin` / `clock_frequency_mhz`; a synchronous claim carries a positive
    clock; and the setup/hold margin fits within the stage's own latency budget (a window can't exceed it).
    R20: a RAW dependency that crosses clock domains -- the producer that last WROTE a read RID declared a
    different `clock_domain` than this consumer -- must be SYNCHRONIZED, i.e. the consumer declares
    `sync_type='mixed'` OR a `barriered` hazard (the synchronizer / handshake, modeled on the existing
    `!bcir.token` fork/await + barrier machinery). An unguarded crossing is a metastability risk."""
    diags: list[Diagnostic] = []
    writer_domain: dict[int, str] = {}  # rid -> clock_domain of the last claim that WROTE it
    for ph in module.phases:
        for claim in ph.claims:
            tm = getattr(claim, "timing", None)
            if tm is not None:
                if tm.sync_type not in _SYNC_TYPES:
                    diags.append(
                        Diagnostic("R19", f"claim {claim.id}: unknown sync_type {tm.sync_type!r}")
                    )
                if tm.latency_cycles < 0:
                    diags.append(
                        Diagnostic(
                            "R19", f"claim {claim.id}: negative latency_cycles {tm.latency_cycles}"
                        )
                    )
                if tm.setup_hold_margin < 0:
                    diags.append(
                        Diagnostic(
                            "R19",
                            f"claim {claim.id}: negative setup_hold_margin {tm.setup_hold_margin}",
                        )
                    )
                if tm.clock_frequency_mhz < 0:
                    diags.append(
                        Diagnostic(
                            "R19",
                            f"claim {claim.id}: negative clock_frequency_mhz {tm.clock_frequency_mhz}",
                        )
                    )
                if tm.sync_type == "synchronous" and tm.clock_frequency_mhz <= 0:
                    diags.append(
                        Diagnostic(
                            "R19",
                            f"claim {claim.id}: a synchronous claim needs a positive clock_frequency_mhz",
                        )
                    )
                if tm.latency_cycles > 0 and tm.setup_hold_margin > tm.latency_cycles:
                    diags.append(
                        Diagnostic(
                            "R19",
                            f"claim {claim.id}: setup_hold_margin {tm.setup_hold_margin} exceeds the "
                            f"stage latency_cycles {tm.latency_cycles}",
                        )
                    )
                synchronized = (tm.sync_type == "mixed") or (claim.hazard == "barriered")
                if tm.clock_domain and not synchronized:
                    for rid in claim.rd:
                        pdom = writer_domain.get(rid)
                        if pdom and pdom != tm.clock_domain:
                            diags.append(
                                Diagnostic(
                                    "R20",
                                    f"claim {claim.id}: reads RID {rid} from clock domain {pdom!r} into "
                                    f"{tm.clock_domain!r} without a synchronizer (declare sync_type='mixed' or "
                                    f"a barriered hazard) -- an unguarded clock-domain crossing",
                                )
                            )
            dom = tm.clock_domain if tm is not None else ""
            for rid in claim.wr:
                writer_domain[rid] = dom
    return diags


_LIFETIME_EVENTS = {"use", "alloc", "free"}


def verify_lifetime(module: Module) -> list[Diagnostic]:
    """R21 (pointer-lifetime legality: use-after-free / double-free) over the OPTIONAL `claim.lifetime`
    annotation -- the naked-pointer safety track (§5.12). A claim may ALLOCATE the resources it writes
    (`event='alloc'`) or FREE the resources it reads (`event='free'`, the `free(p)` shape); every other
    claim is an implicit use. Walking the phase/claim order with the set of currently-FREED resources:

      * a READ of a freed-and-not-reallocated resource is a USE-AFTER-FREE (a dangling dereference: `*p` /
        `p[i]` / `*p = x` all read the pointer to get the address);
      * a `free` of an already-freed resource is a DOUBLE-FREE;
      * a WRITE to a resource RE-VALIDATES it -- `p = malloc(...)` after `free(p)` is legal (reassigning the
        pointer), whereas dereferencing it is not; an explicit `alloc` is the same re-validation.

    Vacuous by construction: with no `free` annotation anywhere (the entire scalar / C-frontend subset
    today, where every claim has `lifetime is None`), nothing is ever freed, so the law emits nothing --
    the non-disturbance invariant, exactly like R19/R20 are vacuous without timing and R17 without a
    tolerance. It becomes load-bearing once a frontend annotates its malloc/free (the lifetime rewrite of
    naked pointers), catching the dangling access the C program would have left UB."""
    diags: list[Diagnostic] = []
    freed: set[int] = set()  # resources freed and not yet re-allocated
    for ph in module.phases:
        for claim in ph.claims:
            lt = getattr(claim, "lifetime", None)
            event = lt.event if lt is not None else "use"
            if lt is not None and event not in _LIFETIME_EVENTS:
                diags.append(
                    Diagnostic("R21", f"claim {claim.id}: unknown lifetime event {event!r}")
                )
                event = "use"
            for rid in claim.rd:  # a READ of a freed resource is the dangling dereference
                if rid in freed:
                    kind = "double-free" if event == "free" else "use-after-free"
                    diags.append(
                        Diagnostic(
                            "R21",
                            f"claim {claim.id}: {kind} of RID {rid} (freed and not re-allocated)",
                        )
                    )
            if event == "free":  # the read resources die after this claim
                for rid in claim.rd:
                    freed.add(rid)
            for rid in claim.wr:  # a WRITE (reassignment / alloc) re-validates the resource
                freed.discard(rid)
    return diags


def verify_shape(module: Module) -> list[Diagnostic]:
    """R22 -- the SHAPE-CONSISTENCY law over `gem.*` tensor claims (D2, ML/AI roadmap §8.2: the
    R19-R21 six-artifact promotion applied to the "structurally valid tensors" guarantee). On the
    model rail a gem claim's tensor rides its written resource, so the checkable structure is the
    producer->consumer SEAM: a gem claim writing a resource that a later gem claim in the same phase
    reads hands over ONE tensor -- both ends must declare the same element extent (`count`). This is
    exactly the adjacency contract the fusion optimizer prices (`matmul_activation`: the activation
    consumes the matmul's full m*n product). Vacuous for a module with no gem seam (the entire
    scalar / C-frontend corpus) -- the non-disturbance invariant, like R19/R20/R21. The MLIR rail
    carries the same law structurally over the gem op shape attrs (`verifyR22` in `-bcir-verify`)."""
    diags: list[Diagnostic] = []
    for ph in module.phases:
        writer: dict[int, Claim] = {}
        for claim in ph.claims:
            if claim.op.startswith("gem."):
                for rid in claim.rd:
                    prod = writer.get(rid)
                    if prod is not None and prod.count != claim.count:
                        diags.append(
                            Diagnostic(
                                "R22",
                                f"claim {claim.id}: gem seam over RID {rid} is shape-inconsistent -- "
                                f"producer claim {prod.id} hands over {prod.count} elements, the "
                                f"consumer declares {claim.count}",
                            )
                        )
            for rid in claim.wr:
                if claim.op.startswith("gem."):
                    writer[rid] = claim
                else:
                    writer.pop(rid, None)  # a non-gem rewrite breaks the tensor seam
    return diags


# Dtype-compatibility (R23) has no structural surface on the Python model rail (a model Claim
# carries no dtype; the activation kind rides the op string) -- the structural R23 law lives on
# the MLIR rail, where the gem ops carry `dtype` attrs (`verifyR23`). The oracle-side R23 surface
# is the SPEC level below (`verify_ml_spec`), which promotes the E3-E6 checkers' dtype rules.
def verify_ml_spec(family: str, errors: list[str]) -> list[Diagnostic]:
    """R22/R23 -- the ML tensor-claim SPEC laws (D2): promotes the op-level
    `check_transformer` / `check_recurrent` / `check_classical` / `check_unsupervised` validators
    to the numbered law surface. A dtype-compatibility message becomes **R23**; every other
    (shape / extent / kind) message becomes **R22**. The checkers themselves stay in `kbcir`
    (cost-side, importing no verifier -- the two-truth quarantine); the CALLER runs the checker
    and passes its messages here, so the sanctioned import direction is preserved and THIS
    function owns the law numbering."""
    return [
        Diagnostic("R23" if ("dtype" in msg or "f32" in msg) else "R22", f"{family}: {msg}")
        for msg in errors
    ]


def verify_barrier_ordering(module: Module, result) -> list[Diagnostic]:
    """ASM3b structural invariant (NOT a verdict R-law): a `barriered`-hazard claim is a first-class
    ordering edge -- no claim may be scheduled across it within its phase. The bundle optimizer
    guarantees this BY CONSTRUCTION (`bundle._conflict` makes a barriered claim conflict with every
    other claim, so `find_bundles` / `_legal_reorder` never move a claim past it); this pass VERIFIES
    the property independently over a realized plan, exactly as `verify_lifetime` (R21) is an advisory
    over the optional metadata -- it is checked OUT of the frontend verdict (`CompileResult.is_clean`),
    never coupled to a legality `ok`.

    Given a `RealizationResult`, for each phase compare the plan's realized claim order against the
    declared (module) order: a barriered claim partitions its phase, so every non-barriered claim must
    stay on its declared side of every barriered claim. A claim that crossed a barrier is flagged
    (the structural breach the bundle.py guard prevents). Vacuous for a plan with no barriered claim
    (the entire deforestation/CSE corpus) -- the non-disturbance invariant."""
    diags: list[Diagnostic] = []
    # The realized claim order PER PHASE, in plan-step order.
    realized: dict[int, list[int]] = {}
    for step in result.steps:
        realized.setdefault(step.phase_id, []).append(step.claim_id)
    for ph in module.phases:
        declared = [c.id for c in ph.claims]
        if not any(c.hazard == "barriered" for c in ph.claims):
            continue  # no fence in this phase: nothing to enforce
        order = realized.get(ph.phase_id)
        if order is None:
            continue  # phase not realized by this plan (nothing to check)
        pos = {cid: i for i, cid in enumerate(order)}
        # Every declared (barriered, other) pair must keep its declared relative order: a barrier is a
        # hard ordering edge, so a claim before it stays before it (and after stays after).
        barriers = [c.id for c in ph.claims if c.hazard == "barriered"]
        for i, a in enumerate(declared):
            for b in declared[i + 1 :]:
                if (a in barriers or b in barriers) and a in pos and b in pos:
                    if pos[a] > pos[b]:
                        diags.append(
                            Diagnostic(
                                "ASM3b",
                                f"phase {ph.phase_id}: claim {a} and {b} reordered across a "
                                f"barriered ordering edge (a barrier is a hard reorder fence)",
                            )
                        )
    return diags


def verify_smart_lowering(
    module: Module, pack=None, dvfs_plan=None, placement=None
) -> list[Diagnostic]:
    """Run the smart-lowering laws R14-R17 over whichever artifacts are provided
    (CIM StreamPack, DVFS plan, allocator placement) plus the always-checkable
    accuracy contract (R17) -- the dual-rail counterpart to the MLIR
    `-bcir-lower-to-llvm` / `-bcir-verify` R14/R15/R16/R17 checks."""
    diags: list[Diagnostic] = []
    if pack is not None:
        diags += verify_cim(pack)
    if dvfs_plan is not None:
        diags += verify_dvfs(dvfs_plan)
    if placement is not None:
        diags += verify_allocator(module, placement)
    diags += verify_accuracy(module)
    diags += verify_timing(module)  # R19/R20 over optional claim.timing -- vacuous without it
    diags += verify_lifetime(module)  # R21 over optional claim.lifetime -- vacuous without it
    diags += verify_shape(module)  # R22 over gem.* tensor seams -- vacuous without them
    return diags


def verify_quarantine(verdict_inputs=(), decisions=(), diagnostics=None) -> list[Diagnostic]:
    """Two-truth quarantine law R13 (MOPC): graded truth may inform a verdict but
    never *be* one. The verifier speaks classical truth (deterministic, binary);
    the learned/measured machinery speaks graded truth `(value, confidence)`. A
    graded proposition may cross into a legality verdict only as the classical
    value of a *recorded* `decide` -- never raw, never silently.

    Flags (all R13):
      * any `verdict_inputs` element that is a graded proposition (or a decision
        still wrapping one) -- graded truth reached a classical boundary uncollapsed;
      * any `decisions` collapse that is malformed (confidence/threshold out of the
        Q-milli range) -- a crossing that cannot be audited;
      * any `diagnostics` carrying a confidence -- a verdict is binary; a graded
        verdict is a category error (the verifier must never emit one).
    """
    from ..kbcir.twotruth import Decision, Graded, is_classical

    diags: list[Diagnostic] = []

    # R13: no graded proposition crosses uncollapsed into a verdict input.
    for x in verdict_inputs:
        if not is_classical(x):
            inner = x.value if isinstance(x, Decision) else x
            conf = getattr(inner, "confidence_milli", "?")
            diags.append(
                Diagnostic(
                    "R13",
                    f"graded proposition (confidence {conf}/1000, source "
                    f"{getattr(inner, 'source', '?')!r}) crossed into a legality "
                    f"verdict; collapse it with decide() first",
                )
            )

    # R13: every recorded collapse is well-formed (auditable).
    for d in decisions:
        if not (0 <= d.confidence_milli <= 1000 and 0 <= d.threshold_milli <= 1000):
            diags.append(
                Diagnostic(
                    "R13",
                    f"malformed two-truth collapse (confidence {d.confidence_milli}, "
                    f"threshold {d.threshold_milli}); a crossing must be auditable",
                )
            )
        if not is_classical(d.value):
            diags.append(
                Diagnostic(
                    "R13",
                    "two-truth collapse still wraps a graded value; decide() must "
                    "yield classical truth",
                )
            )

    # R13: a verdict is classical -- a Diagnostic may not carry a confidence.
    for diag in diagnostics or ():
        if isinstance(diag, Graded) or hasattr(diag, "confidence_milli"):
            diags.append(
                Diagnostic(
                    "R13",
                    "a legality verdict carries a confidence; classical truth is "
                    "binary (graded verdicts are quarantined out of the laws)",
                )
            )

    return diags


def verify_enriched(operad, root=None) -> list[Diagnostic]:
    """Enriched-operad integrity law R13 (the higher memory interface): the
    interpretive layer over the memory modules must be self-consistent so it can
    be trusted to *inform* (never legislate). Witnesses, for a
    `kbcir.operad.EnrichedOperad`:

      * **label consistency** -- when labeling is active, every operation carries a
        non-empty hierarchical label;
      * **content-addressed index integrity** -- when indexing is active, every
        operation's index equals `f_index(name, label, children)` (so identical
        operations share an index: CSE / the liked-pair identity, and tamper is
        evident);
      * **mapping integrity** -- every child index resolves (`Trace` never dangles),
        and, if `root` is given, the whole reachable structure is well-formed.

    This is the analog of `verify_memory` for the enriched structure. The labels
    and indexes it guards are interpretive metadata, quarantined out of R1-R12
    (§14): this law checks the memory interface's own integrity, not legality.
    """
    diags: list[Diagnostic] = []
    for index, reason in operad.problems():
        diags.append(Diagnostic("R13", f"enriched operation {index}: {reason}"))
    if root is not None and operad.get(root) is None:
        diags.append(Diagnostic("R13", f"enriched operad root {root} does not resolve"))
    return diags


def verify_provenance(
    portfolio,
    certificates=(),
    h=None,
    table=None,
    verdicts=(),
    gate_certificates=(),
    accel_certificates=(),
    amortization_certificates=(),
    memory_modules=(),
) -> list[Diagnostic]:
    """Policy/table provenance law R13: every decision rule in force carries a
    generation tag and an admitting certificate -- rule swaps and table
    applications are never silent -- and every boundary verdict carries an MDL
    justification consistent with the recommendation it makes.

    `portfolio` is a `kbcir.portfolio.PolicyPortfolio`, `certificates` an
    iterable of `ReplayCertificate`s, `h` a `TargetProfile`, `table` a
    `kbcir.microbench.CalibratedProfile`, `verdicts` an iterable of
    `kbcir.regret.BoundaryVerdict`, `gate_certificates` the
    `ReplayCertificate`s of learned MoE gates proposed for deployment, and
    `memory_modules` the `kbcir.memory.MemoryModule`s proposed for freezing into
    a generation (all duck-typed).
    """
    diags: list[Diagnostic] = []

    # R13: memory-module provenance -- an e-graph extraction may be frozen and
    # generation-tagged only when it is the resolution fixpoint a = Lim(Res(U))
    # (saturated), never a budget cutoff. The saturation witness is its admitting
    # certificate; this ties the e-graph engine (Phase 21) to the manifest (20).
    for mm in memory_modules:
        diags += verify_memory(mm)

    # R13: learned-gate provenance -- a MoE gate (kbcir.moegate) may deploy only
    # behind an admitting replay certificate (zero regressions vs the incumbent
    # classify router). The network proposes a route; the verifier disposes.
    for cert in gate_certificates:
        if not cert.admitted:
            diags.append(
                Diagnostic(
                    "R13",
                    f"learned gate {cert.candidate!r} cannot deploy: replay gate "
                    f"not passed ({cert.regressions} regression(s) over "
                    f"{cert.episodes} episode(s) vs {cert.incumbent!r})",
                )
            )

    # R13: search-accelerator provenance -- a learned candidate ordering
    # (kbcir.accel) speeds the exact search but must reproduce the exact optimum.
    # A deployed accelerator carries an equivalence certificate with zero
    # mismatches; ordering changes work, never the result.
    for cert in accel_certificates:
        if not cert.admitted:
            diags.append(
                Diagnostic(
                    "R13",
                    f"search accelerator {cert.order_name!r} changed the optimum: "
                    f"{cert.mismatches} mismatch(es) over {cert.checked} case(s)",
                )
            )

    # R13: amortization provenance (the L1 cost throttle) -- a deployed learned
    # component must belong at its tier: the L0 prohibition (no inference on the
    # hot path), it pays for itself (gain >= inference_cost), and it is within the
    # tier's inference budget. Performance is throttled where it matters.
    for cert in amortization_certificates:
        if cert.tier == "L0" and cert.inference_cost != 0:
            diags.append(
                Diagnostic(
                    "R13",
                    f"component {cert.component!r} at L0 runs learned inference "
                    f"(cost {cert.inference_cost}); the hot path carries decisions, "
                    f"not models",
                )
            )
        elif not cert.admitted:
            diags.append(
                Diagnostic(
                    "R13",
                    f"component {cert.component!r} fails amortization at {cert.tier}: "
                    f"gain {cert.gain} vs inference_cost {cert.inference_cost}, "
                    f"budget {cert._budget()}",
                )
            )

    # R13: boundary-verdict provenance -- a retune recommendation must be backed
    # by description-length evidence (data_fit > complexity), and a keep must not
    # sit on regret that has already outgrown the model-complexity cost. The
    # dashboard cannot recommend a swap the MDL/evidence does not justify.
    for v in verdicts:
        justified = v.data_fit_nats > v.complexity_nats
        if v.verdict == "retune" and not justified:
            diags.append(
                Diagnostic(
                    "R13",
                    f"retune verdict for {v.rule!r} is unjustified: data_fit "
                    f"{v.data_fit_nats:.3f} <= complexity {v.complexity_nats:.3f}",
                )
            )
        if v.verdict == "keep" and justified:
            diags.append(
                Diagnostic(
                    "R13",
                    f"keep verdict for {v.rule!r} ignores justified regret: data_fit "
                    f"{v.data_fit_nats:.3f} > complexity {v.complexity_nats:.3f}",
                )
            )

    # R13: gain-schedule provenance -- a promoted (gen > 1) certified entry must
    # be backed by an admitting replay certificate covering exactly that swap.
    if portfolio is not None:
        for slot, entry in sorted(portfolio.entries.items()):
            # "Every decision rule in force carries a generation tag" -- certified or not (the
            # law rail refuses any gen < 1 entry; S0-6 aligns the oracle to that one rule).
            if entry.gen < 1:
                diags.append(Diagnostic("R13", f"policy {slot!r} has no generation tag"))
            # A promotion is witnessed or it did not happen: every gen > 1 entry, certified or
            # not, needs the replay certificate that minted its generation (S0-6: the law rail
            # already held every entry to this; the oracle skipped uncertified ones).
            if entry.gen > 1:
                covering = [
                    c
                    for c in certificates
                    if c.admitted and c.candidate == entry.policy.name and c.incumbent == slot
                ]
                if not covering:
                    diags.append(
                        Diagnostic(
                            "R13",
                            f"promoted policy {slot!r} (gen {entry.gen}) has no "
                            f"admitting replay certificate",
                        )
                    )

    # R13: cost-table provenance -- a profile claiming calibration must present
    # the certified table it was calibrated under, with matching generation and
    # constants (no silent drift between the table and the rule in force).
    if h is not None and getattr(h, "cal_gen", 0) >= 1:
        if table is None:
            diags.append(
                Diagnostic(
                    "R13",
                    f"profile {h.name!r} claims cal_gen {h.cal_gen} but presents "
                    f"no certified table",
                )
            )
        elif table.cal_gen != h.cal_gen:
            diags.append(
                Diagnostic(
                    "R13",
                    f"stale calibration: profile cal_gen {h.cal_gen} != table "
                    f"cal_gen {table.cal_gen} (recalibrate or rehydrate)",
                )
            )
        elif (h.gather_penalty, h.base_overhead, h.mem_unit) != (
            table.gather_penalty,
            table.base_overhead,
            table.mem_unit,
        ):
            diags.append(
                Diagnostic(
                    "R13",
                    f"profile constants drifted from the certified table (cal_gen {h.cal_gen})",
                )
            )

    # R13: conformal-guarantee provenance -- a Bayesian-calibrated table that
    # claims a coverage level must carry a valid coverage in (0,1) and a
    # non-negative +/- delta, and it may not certify a finite interval from too
    # few samples (a coverage guarantee from <= 1 observation is not a guarantee).
    if table is not None and getattr(table, "coverage_milli", 0):
        cov = table.coverage_milli
        delta = getattr(table, "random_delta_q8", 0)
        samples = getattr(getattr(table, "point", None), "samples", 0)
        if not (0 < cov < 1000):
            diags.append(Diagnostic("R13", f"conformal coverage {cov}/1000 out of range (0,1000)"))
        if delta < 0:
            diags.append(Diagnostic("R13", "conformal delta must be >= 0"))
        if delta > 0 and samples <= 1:
            diags.append(
                Diagnostic(
                    "R13",
                    f"conformal interval (delta {delta}) certified from {samples} "
                    f"sample(s): a coverage guarantee needs >= 2 observations",
                )
            )

    return diags


def verify_all(
    module: Module,
    result=None,
    pack=None,
    ll_text: str | None = None,
    elem: str = "f32",
    width_override: int | None = None,
    h=None,
    theta=None,
    policy=None,
    budget=None,
) -> list[Diagnostic]:
    """Run the R1-R12 chain (and EV1-EV3) over every artifact provided (R13 attaches to
    the decision rules, not the module: see `verify_provenance`).

    `h`, `theta`, `policy` and `budget` reach R9's re-derivation; see `verify_plan`. Pass
    whatever the caller planned with -- a chain verified without its scope has not
    re-derived the offer, the costs or the budget."""
    diags = verify(module)
    if result is not None:
        diags += verify_plan(module, result, h, theta=theta, policy=policy, budget=budget)
    if pack is not None:
        diags += verify_pack(module, pack, result)
    if ll_text is not None and result is not None:
        diags += verify_lowering(module, result, ll_text, elem, width_override)
    return diags


def is_legal(module: Module) -> bool:
    return not verify(module)


def _conflict(a, b) -> bool:
    """A read/write hazard between two claims (RAW / WAR / WAW)."""
    aw, ar = set(a.wr), set(a.rd)
    bw, br = set(b.wr), set(b.rd)
    return bool(aw & (br | bw)) or bool(bw & ar)


def _is_sparse(c) -> bool:
    return c.lane == Lane.GGG or c.stride_class == StrideClass.RANDOM


def _topo_phase_ids(module: Module) -> list[int]:
    return topological_phase_ids(module)


def _has_cycle(module: Module) -> bool:
    return phase_graph_has_cycle(module)
