"""Emit -> re-parse ROUND-TRIP IDEMPOTENCE for the cfront C backend -- the missing fixed-point gate.

cfront emits C (`bcir_<fn>(...)`). Elsewhere that emitted C is validated only by COMPILING it
(`test_c_cfront`'s Clang behaviour-equivalence). This module closes the other half: that the emit is a
STABLE FIXED POINT of the parse->lower->emit pipeline -- once a program is in the emitted-C sublanguage,
re-parsing + re-emitting it does not drift the claim graph. A bug where the emit is valid C but
re-lowers to a DIFFERENT graph (a hidden emitter asymmetry) is invisible to a compile-only check; it is
exactly what a round-trip surfaces. This gate is PURE-PYTHON (no C compiler) -- it runs in every tier.

    e1 = emit(parse(src))      # the emitted C, per function, joined (attestation comment stripped)
    g2 = parse(e1)             # re-parse the emitted C through cfront
    e2 = emit(g2)              # re-emit
    g3 = parse(e2)             # re-parse again

The invariant gated is the SUMMARY-LEVEL FIXED POINT (the strongest that honestly holds; see below):

    observable_signature(g2) == observable_signature(g3)

i.e. the per-function multiset of OBSERVABLE claim ops is a fixed point from the emit onward.

WHY NOT byte-idempotence (`e2 == e1`).  It holds for ZERO fixtures -- the emit is structurally not a
retraction, for two benign + well-understood reasons:
  1. the emitter always prepends `bcir_`, so a re-emit renames `bcir_f` -> `bcir_bcir_f` (the rename
     ACCUMULATES across rounds); and
  2. the emit declares each intermediate as a single-assignment temporary (`uint32_t t104 = expr;`), but
     re-parsing materializes every read-back temporary as a NAMED MUTABLE LOCAL (declared up front +
     assigned via a `c.copy`), so the claim graph gains `c.copy` book-keeping each round and the temp
     numbering shifts. The TOTAL `claims=` count therefore grows monotonically round over round -- so
     neither byte-idempotence NOR the full production-summary string is ever a fixed point.
The observable signature projects those two book-keeping sources out (it drops `c.copy`, drops the loop
`while(1)` scaffold `c.const`, and normalizes the `bcir_` call-rename), leaving the semantically
load-bearing ops (arithmetic / loads / stores / library + indirect calls / bitfields / casts / selects /
label addresses). On that projection the round trip IS a genuine fixed point -- a real, non-tautological
property, since the parser + emitter run a SECOND time on the emitter's own output and any op-graph
asymmetry (an op dropped, doubled, or retyped on re-lowering) breaks it.

EXCLUDED fixtures (classified + pinned, exactly like a fairness gate -- a regression that GROWS the set
is visible).  A fixture is excluded, never forced, when its emit legitimately leaves the re-parseable /
idempotent subset:
  * "emit-not-reparseable": the standalone emit references a name only the ORIGINAL source defined -- an
    aggregate/enum type (`struct uart_regs`), a file-scope `#define` constant (`GAMMA`), an external
    macro (`__ATOMIC_SEQ_CST`), or an in-body VLA whose extent the lowering rejects standalone. The emit
    is faithful C in context but not a self-contained translation unit, so cfront cannot re-lower it.
  * "control-flow-not-idempotent": the emit lowers a structured loop to `while (1) { ...; if (!cond)
    break; ... }`. Re-parsing that re-introduces the `while(1)` wrapper + a fresh negation each round
    (the `c.un.lnot` break-test count grows), and the emitter's `t<rid>` temp names collide with a
    re-parsed local that happens to spell `t<NN>` -- so the SECOND-round emit does not even compile. The
    loop scaffold is genuinely not a fixed point; pinned with this reason rather than papered over.
  * "masked-guard-not-idempotent": the emit of a `masked` (§5.12 bounds-promoted) access is
    `a[BCIR_CHK(rid, i, n, "site")]`. `BCIR_CHK` is a runtime macro absent from the standalone emit, so a
    re-parse reads it as a `c.call:BCIR_CHK`, and a re-emit re-wraps the index -- the guard count grows
    each round. Outside the idempotent subset.

This classification is DISCOVERED dynamically (re-parse + inspect), and the gate pins the resulting
counts, so a regression that shrinks the included set or grows the excluded set fails the smoke test.

Mirrors the round-robin slice pattern of `test_c_cfront._PARITY_SLICES`: N independent `test_*` groups
run_all fans out, plus a non-skippable anti-degeneration smoke test.
"""

import re

from bcir.frontends.cfront import compile_unit

# Reuse the canonical fixture corpus + the header-resolution helper from the C-twin parity module, so a
# fixture added there is auto-covered here (no second hand-maintained corpus to drift).
from bcir.tests.test_c_cfront import _ATOMIC, _C, _FIXTURES, _includes_for  # noqa: PLC0415

# The whole shared corpus (straight-line + control-flow + ABI + float + atomics), in a stable order.
_CORPUS = _FIXTURES + _ATOMIC

# The attestation comment the pipeline staples onto each emitted function (Phase C.2 provenance). It is a
# generated banner, not part of the lowered program, so it is stripped before re-parsing + comparing.
_ATTESTATION = re.compile(r"/\* BCIR verified-C-output attestation.*?\*/\n?", re.DOTALL)

# The two book-keeping claim ops the round trip introduces that carry NO observable semantics (see the
# module docstring): a `c.copy` materializes an SSA temp as a named local; a `c.const` re-appears as the
# `1u` of a re-lowered `while (1)` loop scaffold. Both are projected out of the observable signature.
_BOOKKEEPING = frozenset({"c.copy", "c.const"})


def _emit_joined(result) -> str:
    """The emitted C for every function in the unit, attestation banners stripped, joined -- the `e1`/`e2`
    of the round trip. The strip makes the text a self-contained C translation unit candidate."""
    return "\n\n".join(_ATTESTATION.sub("", result.emitted[name])
                       for name in result.lowered.functions)


def _norm_op(op: str) -> str:
    """Normalize a claim op for the observable signature: strip the accumulating `bcir_` rename prefix
    from a DIRECT-call callee (`c.call:l4_scale` <-> `c.call:bcir_l4_scale`) so the rename -- which the
    emitter applies unconditionally and which therefore accumulates across rounds -- does not register as
    a graph change. The op family and the call COUNT are untouched; only the callee spelling is canonical."""
    if op.startswith("c.call:") or op.startswith("c.call.void:"):
        head, callee = op.split(":", 1)
        return head + ":" + re.sub(r"^(bcir_)+", "", callee)
    return op


def _observable_signature(result) -> tuple:
    """The per-function multiset of OBSERVABLE claim ops (book-keeping projected out, the `bcir_` call
    rename normalized). The semantically load-bearing fingerprint of the lowered unit -- a fixed point of
    the round trip iff the emit re-lowers to the same observable graph. Position-keyed (function order),
    so it does not depend on any name (which the `bcir_` rename perturbs)."""
    sig = []
    for name in result.lowered.functions:
        kinds: dict[str, int] = {}
        for c in result.lowered.functions[name].claims:
            if c.op in _BOOKKEEPING:
                continue
            o = _norm_op(c.op)
            kinds[o] = kinds.get(o, 0) + 1
        sig.append(tuple(sorted(kinds.items())))
    return tuple(sig)


def _reparse(emit_text: str):
    """Re-lower the emitted C through cfront, or None if it leaves the re-parseable subset (it references
    a name only the original source defined, or a construct the standalone lowering rejects). A unit that
    parses but does not verify clean (`is_clean` false) is also treated as not re-parseable -- the round
    trip is defined only over units cfront fully accepts."""
    try:
        r = compile_unit(emit_text, check_clang=False)
    except Exception:  # noqa: BLE001 -- any front-end stage rejecting the standalone emit -> excluded
        return None
    return r if r.is_clean else None


def _nonidempotent_construct(emit_text: str) -> str | None:
    """A statically-detectable emit construct known to leave the IDEMPOTENT subset (vs the re-PARSEABLE
    subset `_reparse` handles). Detected by inspecting `e1` directly, so the reason is precise and does
    not depend on the failure mode of a second round:
      * `while (1)`  -- the structured-loop scaffold (`while(1){...; if(!cond) break; ...}`) re-lowers,
                        growing the break-negation each round (and the 2nd-round emit fails to compile);
      * `BCIR_CHK`   -- a §5.12 masked-access guard macro, re-parsed as a call and re-wrapped each round."""
    if "while (1)" in emit_text:
        return "control-flow-not-idempotent"
    if "BCIR_CHK" in emit_text:
        return "masked-guard-not-idempotent"
    return None


# The pinned floor on how many fixtures must actually exercise the round trip (the anti-degeneration
# guard). Discovery currently includes 62 of the 136-fixture corpus; the floor is set comfortably below
# so a benign corpus reshuffle does not trip it, but a wholesale collapse (a regression that pushes the
# whole corpus into FALLBACK, say) does. `test_roundtrip_smoke` pins the exact split.
_MIN_EXERCISED = 50


def _classify(fx: str) -> tuple[str, str]:
    """Round-trip classification of ONE fixture. Returns `(verdict, detail)`:
      * ("included", "")             -- e1 re-parses, contains no non-idempotent construct, and the
                                        observable signature is a FIXED POINT (osig(g2) == osig(g3));
      * ("drift", "<diag>")          -- e1 re-parses cleanly but the observable signature DRIFTS across
                                        the round trip (osig(g2) != osig(g3)). This is the failure the
                                        gate exists to catch -- a genuine emit<->parse asymmetry; the
                                        slice tests fail on it.
      * ("excluded", "<reason>")     -- the emit legitimately leaves the re-parseable / idempotent subset
                                        (see the classified reasons in the module docstring)."""
    src = open(f"{_C}/{fx}", encoding="utf-8").read()
    try:
        r1 = compile_unit(src, check_clang=False, includes=_includes_for(fx))
    except Exception as e:  # noqa: BLE001 -- a fixture the oracle itself can't lower is out of scope here
        return ("excluded", f"original-not-lowerable:{type(e).__name__}")
    if not r1.is_clean:
        return ("excluded", "original-not-clean")
    e1 = _emit_joined(r1)
    construct = _nonidempotent_construct(e1)
    if construct:
        return ("excluded", construct)
    g2 = _reparse(e1)
    if g2 is None:
        return ("excluded", "emit-not-reparseable")
    e2 = _emit_joined(g2)
    g3 = _reparse(e2)
    if g3 is None:
        return ("excluded", "emit2-not-reparseable")
    s2, s3 = _observable_signature(g2), _observable_signature(g3)
    if s2 != s3:
        return ("drift", f"observable signature drifted across the round trip\n"
                         f"     parse(e1): {s2}\n     parse(e2): {s3}")
    return ("included", "")


# Round-robin slices: 4 independent `test_*` groups run_all's worker pool fans out (the same shape as
# `test_c_cfront._PARITY_SLICES`), so any fixture added to the corpus is auto-covered with no
# hand-maintained partition.
_GROUPS = 4
_SLICES = [_CORPUS[i::_GROUPS] for i in range(_GROUPS)]


def _check_slice(fxs):
    """Round-trip fixed-point check over a SLICE of the corpus. For every fixture whose emit is in the
    re-parseable + idempotent subset, assert the observable claim-op signature is a FIXED POINT across the
    emit -> re-parse -> re-emit -> re-parse round trip. A fixture outside that subset is classified +
    excluded (never forced). A genuine `drift` verdict FAILS the slice (that is the bug this gate exists
    to surface). Pure-Python -- no compiler, so it runs in every tier."""
    drifts, exercised = [], 0
    for fx in fxs:
        verdict, detail = _classify(fx)
        if verdict == "drift":
            drifts.append(f"{fx}: {detail}")
        elif verdict == "included":
            exercised += 1
    assert not drifts, ("cfront emit -> re-parse round-trip NOT a fixed point (the emitted C re-lowers to "
                        "a DIFFERENT observable claim graph -- an emit<->parse asymmetry):\n"
                        + "\n".join(f"  {m}" for m in drifts))
    # Anti-degeneration: a slice that excluded EVERY fixture proves nothing. Each slice must round-trip at
    # least one fixture, so the gate cannot pass by classifying all work away.
    assert exercised > 0, (f"round-trip exercised NO fixture in this slice -- the gate degenerated to all-"
                           f"excluded (a vacuous pass): {fxs}")


def test_emitted_c_reparse_idempotent_g0():
    """Emit -> re-parse observable-signature fixed point, group 0/4 (see `_check_slice`)."""
    _check_slice(_SLICES[0])


def test_emitted_c_reparse_idempotent_g1():
    """Emit -> re-parse observable-signature fixed point, group 1/4 (see `_check_slice`)."""
    _check_slice(_SLICES[1])


def test_emitted_c_reparse_idempotent_g2():
    """Emit -> re-parse observable-signature fixed point, group 2/4 (see `_check_slice`)."""
    _check_slice(_SLICES[2])


def test_emitted_c_reparse_idempotent_g3():
    """Emit -> re-parse observable-signature fixed point, group 3/4 (see `_check_slice`)."""
    _check_slice(_SLICES[3])


def test_roundtrip_is_genuinely_a_second_lowering():
    """A non-skippable proof the gate actually RE-PARSES the emitted C (not a tautology): on a
    representative straight-line fixture, the emitted C must be a DIFFERENT byte string from the source
    (so a real parse happened, not an identity), yet its observable signature is a fixed point from the
    emit onward. Pins, by direct demonstration, that byte-idempotence does NOT hold (the `bcir_` rename +
    SSA-temp -> local materialization) while the observable fixed point DOES -- the exact reason the gate
    uses the projected invariant rather than `e2 == e1`."""
    fx = "cfront_logic.c"                                        # straight-line, no control flow / masked access
    src = open(f"{_C}/{fx}", encoding="utf-8").read()
    r1 = compile_unit(src, check_clang=False, includes=_includes_for(fx))
    assert r1.is_clean, fx
    e1 = _emit_joined(r1)
    g2 = _reparse(e1)
    assert g2 is not None, f"{fx}: emitted C did not re-parse cleanly"
    e2 = _emit_joined(g2)
    g3 = _reparse(e2)
    assert g3 is not None, f"{fx}: re-emitted C did not re-parse cleanly"
    # A real re-parse happened: the re-emit is NOT byte-identical (the `bcir_` rename + SSA-temp->local
    # materialization), so the fixed point is a genuine claim-graph property, not a string identity.
    assert e2 != e1, f"{fx}: emit was byte-identical -- the round trip would be a tautology"
    assert "bcir_bcir_" in e2, f"{fx}: expected the accumulating bcir_ rename in the re-emit"
    # ...and the OBSERVABLE claim-op signature is a fixed point from the emit onward.
    assert _observable_signature(g2) == _observable_signature(g3), (
        f"{fx}: observable signature drifted across the round trip "
        f"({_observable_signature(g2)} -> {_observable_signature(g3)})")


def test_roundtrip_smoke_and_exclusion_set_is_pinned():
    """The anti-degeneration + excluded-set pin (the gate's teeth). Classifies the WHOLE corpus once and
    asserts:
      * NO fixture drifts (a `drift` verdict anywhere is a real emit<->parse asymmetry -> hard fail);
      * a non-trivial number of fixtures actually exercised the round trip (>= the pinned floor), so the
        gate cannot pass by excluding everything;
      * every excluded fixture carries one of the KNOWN classified reasons (an unrecognized reason -- e.g.
        a new fallback path -- fails, surfacing a change in the re-parseable subset);
      * the included/excluded COUNTS are pinned, so a regression that shrinks coverage is visible.
    This is the round-trip analog of `test_c_cfront`'s differential fairness/anti-skip guards."""
    included, excluded, drifted = [], {}, []
    for fx in _CORPUS:
        verdict, detail = _classify(fx)
        if verdict == "included":
            included.append(fx)
        elif verdict == "drift":
            drifted.append(f"{fx}: {detail}")
        else:
            excluded[fx] = detail
    assert not drifted, ("cfront emit -> re-parse round-trip drift (a real emitter asymmetry):\n"
                         + "\n".join(f"  {m}" for m in drifted))
    # the anti-degeneration floor: a large, non-trivial slice of the corpus round-trips.
    assert len(included) >= _MIN_EXERCISED, (
        f"round trip exercised only {len(included)} fixtures (floor {_MIN_EXERCISED}) -- the included set "
        f"collapsed; coverage regression. included={sorted(included)}")
    # every exclusion must carry a KNOWN classified reason -- a new/unrecognized reason fails, so a change
    # to the re-parseable subset (a new fallback path, a renamed reason) is caught rather than absorbed.
    _KNOWN = ("emit-not-reparseable", "emit2-not-reparseable", "control-flow-not-idempotent",
              "masked-guard-not-idempotent", "original-not-clean", "original-not-lowerable:")
    bad = {fx: why for fx, why in excluded.items() if not any(why.startswith(k) for k in _KNOWN)}
    assert not bad, f"excluded fixtures with an UNRECOGNIZED reason (the re-parseable subset changed): {bad}"
    # the whole corpus is accounted for (included + excluded == corpus), and the counts are pinned so a
    # coverage regression (a fixture sliding from included into excluded) is visible in the diff.
    assert len(included) + len(excluded) == len(_CORPUS), (len(included), len(excluded), len(_CORPUS))
    assert len(included) == 62, (f"included-set size changed from the pinned 62 to {len(included)} -- a "
                                 f"fixture moved across the round-trip boundary; re-classify + re-pin. "
                                 f"included={sorted(included)}")
