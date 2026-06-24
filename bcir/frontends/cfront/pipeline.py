"""The C-frontend pipeline — the six-artifact gate the ladder requires, end to end:

    C source ── parse ──▶ claim graph ── K_BCIR ──▶ plan ── emit ──▶ verified C ── clang ──▶ ≡ check
                                  └────────────── R1–R18 verifier checkpoint ──────────────┘

`compile_unit` returns every artifact for a translation unit: the parsed C, the lowered claim graph
+ K_BCIR plan + `bcir-explain` text per function, the emitted C, the R1–R18 diagnostics (R18 via the
real `plan_composite` call-graph machinery), and the Clang behaviour-equivalence verdict (which skips
cleanly without a C compiler, so the structural artifacts still run in the quick tier).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, replace

from ...channels import host_channel
from ...kbcir.compose import Effect, plan_composite, summarize
from ...kbcir.cost import Theta
from ...kbcir.realize import optimize
from ...kbcir.weights import PERF
from ...verify import Diagnostic, verify, verify_lifetime, verify_plan
from .abi import HOST, target as resolve_target
from .clex import CLexError
from .cparse import CParseError, parse_unit, parse_with_recovery
from .cpp import CPPError, preprocess
from .diagnostics import DiagnosticReport, SourceDiagnostic, Span
from .emit import emit_function
from .lower import CLowerError, LoweredUnit, lower_unit, own_footprint


# module-scope resource rids (file-scope globals + string-literal globals; see lower_unit) -- the
# only state shared *between* functions, so the alias/effect footprint is restricted to these for the
# inter-procedural commutation analysis (per-function local temps live in disjoint rid ranges).
_SHARED_RID = 900000


def _diag_from(e: Exception, phase: str) -> SourceDiagnostic:
    """A front-end exception -> a located source diagnostic (a caret span when the exception carries a
    `pos`, else a file-level banner)."""
    pos = getattr(e, "pos", None)
    span = Span.at(pos) if pos is not None else None
    return SourceDiagnostic("error", str(e), span=span, phase=phase)


@dataclass
class CompileResult:
    source: str
    unit: object
    lowered: LoweredUnit
    plans: dict = field(default_factory=dict)         # fn -> RealizationResult
    emitted: dict = field(default_factory=dict)       # fn -> C text
    explain: dict = field(default_factory=dict)       # fn -> bcir-explain text
    attestation: dict = field(default_factory=dict)   # fn -> R12/R13/R17/R18 attestation dict
    effects: dict = field(default_factory=dict)        # fn -> Effect (global read/write footprint)
    ipo: dict = field(default_factory=dict)            # inter-procedural plan: worst/expected/leaves/reused
    diagnostics: list = field(default_factory=list)   # R1–R18 Diagnostics (empty == clean)
    lifetime_diagnostics: list = field(default_factory=list)  # R21 use-after-free / double-free (§5.12) --
    #     ADVISORY: a smart-lowering law over the optional `claim.lifetime` (malloc/free), NOT part of the
    #     frontend pass/fail (`is_clean`), exactly as R19/R20 timing are advisory. Empty == no dangling access.
    r18_ok: bool = True
    equivalence: str = "skip"                         # match | MISMATCH | skip:<reason>
    target: str = HOST.name                           # the target ABI the unit was laid out for
    fallback: str = ""                                 # non-empty == BCIR can't compile this; route to LLVM

    @property
    def is_clean(self) -> bool:
        return not self.diagnostics and self.r18_ok and not self.fallback

    @property
    def needs_fallback(self) -> bool:
        """The LLVM-backend fallback signal: True iff BCIR could not compile this unit (a construct
        outside the supported subset, or a malformed unit) and the driver should hand it to LLVM."""
        return bool(self.fallback)

    @property
    def behaviour_equivalent(self) -> bool:
        return self.equivalence == "match"

    def commute(self, a: str, b: str) -> bool:
        """Whether functions `a` and `b` are independent -- their alias/effect footprints over shared
        (module-scope) state do not conflict (no RAW/WAR/WAW), so the inter-procedural scheduler may
        reorder or overlap them. Two readers of the same global commute; a writer does not."""
        return not self.effects[a].conflicts(self.effects[b])


def _cc():
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def diagnose(source: str, *, includes: dict | None = None, embeds: dict | None = None,
             search_paths: list | None = None, defines: dict | None = None,
             filename: str = "<source>") -> DiagnosticReport:
    """Run the front end (preprocess -> parse -> lower) for diagnostics only, returning a
    `DiagnosticReport`: the source diagnostics plus the text their caret spans index into. An empty
    report (`ok`) means the unit is well-formed through lowering. A front-end error is caught and
    rendered with a Clang-style caret at its source span instead of crashing the caller.

    The parser recovers in panic mode, so a single run reports *every* syntax error it can resynchronize
    past (not just the first). Preprocess and lex errors are still one-shot; semantic (lowering)
    diagnostics report the first, and only run once the parse is clean."""
    try:
        pp, line_map = preprocess(source, includes=includes, embeds=embeds, search_paths=search_paths,
                                  defines=defines, name=filename, return_map=True)
    except CPPError as e:
        return DiagnosticReport([_diag_from(e, "preprocess")], source, filename)
    try:
        unit, parse_diags = parse_with_recovery(pp)   # panic-mode recovery: collect all syntax errors
    except CLexError as e:                            # a lexer error is not recovered (separate concern)
        return DiagnosticReport([_diag_from(e, "lex")], pp, filename, line_map)
    if parse_diags:                                   # syntax errors -> report them all; AST is partial
        return DiagnosticReport(parse_diags, pp, filename, line_map)
    try:
        lower_unit(unit)                              # only lower a syntactically clean unit
    except CLowerError as e:
        return DiagnosticReport([_diag_from(e, "lower")], pp, filename, line_map)
    return DiagnosticReport([], pp, filename, line_map)


def compile_unit(source: str, *, includes: dict | None = None, embeds: dict | None = None,
                 search_paths: list | None = None, defines: dict | None = None,
                 check_clang: bool = True, filename: str = "<source>",
                 target: str | None = None) -> CompileResult:
    # L7: preprocess first — the expanded text is what both the parser and the Clang harness see,
    # so the equivalence check validates the lowering of the *preprocessed* program. `search_paths`
    # (the source dir + -I dirs) resolves `#include "..."` from disk; `defines` seeds -D macros;
    # `filename` is the translation unit's name for __FILE__. `target` selects the ABI the unit lays
    # out for (defaults to the host); a non-host layout is not byte-compatible with the host clang, so
    # its behaviour-equivalence check is skipped (the layout is conformance-checked instead).
    abi = resolve_target(target) if target else HOST
    source = preprocess(source, includes=includes, embeds=embeds,
                        search_paths=search_paths, defines=defines, name=filename)
    unit = parse_unit(source)
    lowered = lower_unit(unit, abi)
    h, theta, policy = host_channel().profile, Theta.cool(), PERF
    res = CompileResult(source=source, unit=unit, lowered=lowered, target=abi.name)

    # --- per-function: plan, verify (R1–R9), emit, explain ---
    for name, lf in lowered.functions.items():
        diags = verify(lf.module)
        plan = optimize(lf.module, h, theta, policy)
        diags += verify_plan(lf.module, plan)
        res.plans[name] = plan
        res.emitted[name] = emit_function(lf)
        res.explain[name] = _explain(lf.module, h, theta, policy)
        res.diagnostics += [Diagnostic(d.law, f"{name}: {d.message}") for d in diags]
        # R21 lifetime (§5.12): an ADVISORY pass over the optional malloc/free `claim.lifetime` -- a
        # use-after-free / double-free a C program would have left UB. Kept OUT of `res.diagnostics`
        # (and thus `is_clean`): like the R19/R20 timing laws, it never changes the frontend verdict.
        res.lifetime_diagnostics += [Diagnostic(d.law, f"{name}: {d.message}")
                                     for d in verify_lifetime(lf.module)]

    # --- alias/effect footprint per function (over shared/module-scope state) -- the basis for
    #     commutation/independence. Each function's own footprint is folded with its callees'
    #     (transitively, recursion-guarded) so a wrapper inherits what it calls. ---
    _own = {name: own_footprint(lf, _SHARED_RID) for name, lf in lowered.functions.items()}

    def _folded(name: str, seen: frozenset) -> tuple:
        if name not in _own or name in seen:
            return frozenset(), frozenset()                # external/opaque callee or a recursion guard
        seen = seen | {name}
        reads, writes = set(_own[name][0]), set(_own[name][1])
        for callee, _actuals in lowered.functions[name].calls:
            cr, cw = _folded(callee, seen)
            reads |= cr
            writes |= cw
        return frozenset(reads), frozenset(writes)

    for name in lowered.functions:
        res.effects[name] = Effect(*_folded(name, frozenset()))

    # --- R18 + IPO: the inter-procedural call graph via plan_composite. Each function is summarized
    #     once (plan-once); a call whose actuals are cost-compatible with the callee's formals reuses
    #     that summary instead of re-planning the body -- `ipo["reused"]` counts the saved plans. ---
    if lowered.entry:
        region = lowered.compose_functions[lowered.entry].region
        summaries = {}
        for fname, cf in lowered.compose_functions.items():
            formal_rids = {rid for _, rid, _ in lowered.functions[fname].params}
            formals = {rid: lowered.resources.get(rid) for rid in formal_rids}
            try:                                          # a function we can't summarize (e.g. it is
                s = summarize(cf, lowered.compose_functions, formals, h, theta, policy)
                # a summary is reusable on its *formals* (the actuals that vary per call site); drop the
                # internal-temp keys summarize folds in from the effect, else no call ever matches.
                summaries[fname] = replace(s, formal_keys=tuple(
                    (rid, key) for rid, key in s.formal_keys if rid in formal_rids))
            except Exception:  # noqa: BLE001 -- recursive) is simply re-planned, never reused
                pass
        try:
            comp = plan_composite(region, lowered.compose_functions, lowered.resources, h, theta,
                                  policy, summaries=summaries)
            res.r18_ok = True
            res.ipo = {"worst": comp.worst_cost, "expected": comp.expected_cost,
                       "leaves": comp.leaves, "reused": comp.reused}
        except Exception as e:  # noqa: BLE001 -- recursion / undefined callee == an R18 violation
            res.r18_ok = False
            res.diagnostics.append(Diagnostic("R18", f"call-graph integrity: {e}"))

    # --- behaviour equivalence vs Clang (clang-gated; host ABI only -- a cross-target layout is not
    #     byte-compatible with the host compiler, so equivalence there would be meaningless) ---
    if check_clang and abi is HOST:
        res.equivalence = _equivalence(source, lowered)
    elif abi is not HOST:
        res.equivalence = f"skip:cross-target:{abi.name}"

    # --- C.2 attestation: stamp each emitted function with its R12/R13/R17/R18 provenance ---
    for name in lowered.functions:
        res.attestation[name] = _attest(res, name, h.name)
        res.emitted[name] = _attestation_comment(res.attestation[name]) + "\n" + res.emitted[name]
    return res


# front-end exception type -> the stage that rejected the unit, for the fallback reason.
_FALLBACK_PHASE = {CPPError: "preprocess", CLexError: "lex", CParseError: "parse",
                   CLowerError: "lower", ValueError: "emit"}


def compile_with_fallback(source: str, **kwargs) -> CompileResult:
    """The LLVM-backend fallback contract: a **total** entry point. If BCIR fully compiles + verifies
    the unit, the result's `fallback` is empty (and `is_clean` can hold). If a construct is outside the
    supported subset -- or the unit is malformed -- it returns a result with `needs_fallback` set and
    `fallback` carrying the rejecting stage + reason, the signal for the driver to route this unit to
    the LLVM backend instead of crashing. (`compile_unit` itself keeps its raise-on-error contract; a
    driver that wants graceful degradation calls this.)"""
    try:
        return compile_unit(source, **kwargs)
    except tuple(_FALLBACK_PHASE) as e:               # a stage outside BCIR's supported subset
        res = CompileResult(source=source, unit=None, lowered=None,
                            target=(kwargs.get("target") or HOST.name))
        res.fallback = f"{_FALLBACK_PHASE[type(e)]}: {e}"
        return res


def _attest(res: CompileResult, fn: str, target: str) -> dict:
    """The verified-C-output provenance for one function: which laws hold, a claim-graph digest
    (R13-style), the integer-exactness (R17), the call-graph integrity (R18), and the plan cost."""
    lf = res.lowered.functions[fn]
    fp = repr([(c.id, c.op, c.opcode.name, c.rd, c.wr, c.imm, c.domain.name) for c in lf.claims])
    digest = hashlib.sha256(fp.encode()).hexdigest()[:16]
    fn_diags = [d for d in res.diagnostics if d.message.startswith(f"{fn}:") or d.law == "R18"]
    eff = res.effects.get(fn)

    def _names(rids):
        return ", ".join(sorted(res.lowered.resources[r].name for r in rids
                                if r in res.lowered.resources)) or "-"

    return {
        "function": fn,
        "target": target,
        "claims": len(lf.claims),
        "R1_R9_module_plan": "clean" if not fn_diags else "DIRTY",
        "R12_lowering_contract": ("attested-by-clang-equivalence" if res.behaviour_equivalent
                                  else res.equivalence),
        "R13_provenance_digest": digest,
        "R17_accuracy": "exact (integer / Q-fixed, 0 ULP)",
        "R18_callgraph_integrity": "clean" if res.r18_ok else "VIOLATION",
        "effect_reads": _names(eff.reads) if eff else "-",      # alias/effect footprint (shared state)
        "effect_writes": _names(eff.writes) if eff else "-",
        "plan_score": str(getattr(res.plans[fn], "score", "?")),
    }


def _attestation_comment(att: dict) -> str:
    body = "\n".join(f" *   {k:<24} {v}" for k, v in att.items())
    return ("/* BCIR verified-C-output attestation (Phase C.2) — generated, do not edit.\n"
            f"{body}\n */")


def emit_selfcheck(result: CompileResult) -> str:
    """A standalone self-checking C program (the generalized C.2 self-check harness): the original
    source + the emitted `bcir_*` functions + a `main` that runs both on seeded-random inputs and
    prints MATCH iff they agree. Compile + run it to re-verify behaviour-equivalence anywhere."""
    entry = result.lowered.functions.get(result.lowered.entry)
    if entry is None:
        return "/* no entry function */\n"
    return _harness_c(result.source, result.lowered, entry)


def _explain(module, h, theta, policy) -> str:
    try:
        from ...kbcir.proof import explain, explain_text  # noqa: PLC0415
        return explain_text(explain(module, h, theta, policy, target_name=h.name))
    except Exception as e:  # noqa: BLE001 -- explain is a best-effort artifact
        return f"(explain unavailable: {e})"


# --- the Clang behaviour-equivalence harness ----------------------------------------------------

def _equivalence(source: str, lowered: LoweredUnit) -> str:
    cc = _cc()
    if not cc:
        return "skip:no-cc"
    entry = lowered.functions.get(lowered.entry)
    if entry is None:
        return "skip:no-entry"
    harness = _harness_c(source, lowered, entry)
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "equiv.c")
        exe = os.path.join(d, "equiv")
        with open(src, "w", encoding="utf-8") as f:
            f.write(harness)
        for std in ("-std=c23", "-std=c2x", "-std=c17"):
            b = subprocess.run([cc, std, "-O1", src, "-o", exe, "-lm"],   # -lm: <math.h> calls link
                               capture_output=True, text=True)
            if b.returncode == 0:
                break
        else:
            return f"skip:build-failed:{b.stderr.strip().splitlines()[-1] if b.stderr else '?'}"
        run = subprocess.run([exe], capture_output=True, text=True)
        out = run.stdout.strip()
        return "match" if out == "MATCH" else f"MISMATCH:{out}"


_BOUNDS_GUARD = (
    "/* §5.12 bounds-quarantine guard, mirrored from runtime/c/bcir_quarantine.h for the in-process\n"
    "   equivalence build: in-bounds -> the raw index (transparent); the handler is a non-aborting stub\n"
    "   (a masked access is provably bounded, so it is never reached). */\n"
    "static size_t bcir_bounds_quarantine(uint64_t rid, uint64_t index, uint64_t extent, const char *site)\n"
    "{ (void)rid; (void)index; (void)extent; (void)site; return 0; }\n"
    "#define BCIR_CHK(rid, i, n, site) \\\n"
    "    ((uint64_t)(i) < (uint64_t)(n) ? (size_t)(i) \\\n"
    "     : bcir_bounds_quarantine((uint64_t)(rid), (uint64_t)(i), (uint64_t)(n), (site)))\n"
)


def _harness_c(source: str, lowered: LoweredUnit, entry) -> str:
    """A self-contained C program: the original source + the emitted bcir_* functions + a main that
    runs both on the same seeded-random inputs and prints MATCH iff every trial agrees."""
    from .emit import _cname  # noqa: PLC0415

    emitted = "\n\n".join(emit_function(lf) for lf in lowered.functions.values())
    # §5.12: a masked (bounds-promoted) array access is emitted as `a[BCIR_CHK(rid, i, n)]`. Mirror the
    # runtime ABI here so the emitted unit compiles+links standalone: BCIR_CHK is transparent in-bounds
    # (so a provably-bounded `masked` access is byte-identical to the raw `a[i]`), and the handler is a
    # non-aborting stub -- a masked access never goes out of bounds, so it is never called in a trial.
    guard = _BOUNDS_GUARD if "BCIR_CHK" in emitted else ""
    has_ptr = any(ct.kind in ("pointer", "array") for _n, _r, ct in entry.params)
    decls, origargs, setup, prelude = [], [], [], []
    for i, (pname, _rid, ct) in enumerate(entry.params):
        if ct.kind == "funcptr":                           # pass a real (deterministic) target fn
            rety = _cname(ct.of) if ct.of else "uint32_t"
            plist = ", ".join(f"{_cname(pt)} p{j}" for j, pt in enumerate(ct.params)) or "void"
            comb = " + ".join(f"(p{j} * {2 * j + 1}u)" for j in range(len(ct.params))) or "1u"
            prelude.append(f"static {rety} _fp{i}({plist}) {{ return ({rety})({comb}); }}")
            origargs.append(f"_fp{i}")
            continue
        if ct.kind in ("pointer", "array"):
            elem = _cname(ct.of)
            decls.append(f"    static {elem} buf{i}[256];")
            # word-fill the backing store (works for scalar AND aggregate/bitfield element types).
            setup.append(f"        for (unsigned k = 0; k < sizeof(buf{i}) / 4; k++) "
                         f"((uint32_t *)buf{i})[k] = rng();")
            origargs.append(f"buf{i}")
        elif ct.is_aggregate:
            decls.append(f"    {ct.kind} {ct.name} a{i};")
            inits = []
            for fname, ftype, _bo, _bf, bw in ct.fields:
                rhs = f"(rng() & {(1 << bw) - 1}u)" if bw else f"({_cname(ftype)})rng()"
                inits.append(f"        a{i}.{fname} = {rhs};")
            setup.append("\n".join(inits))
            origargs.append(f"a{i}")
        else:                                              # scalar; keep small when indexing memory
            decls.append(f"    {_cname(ct)} s{i};")
            # a float param gets an in-range value (so a float->int cast stays defined, not UB); an
            # integer scalar stays below 2**31 so it is non-negative as `int` (the value model is
            # unsigned -- an int->float cast must agree in sign; wrapping arithmetic is unaffected).
            mod = 1000 if ct.is_float else (200 if has_ptr else 2000000000)
            setup.append(f"        s{i} = ({_cname(ct)})(rng() % {mod});")
            origargs.append(f"s{i}")
    call = ", ".join(origargs)
    # struct-by-value returns (L8 ABI) can't be `!=`-compared — diff the bytes via memcmp.
    if entry.ret_type.is_aggregate:
        rt = _cname(entry.ret_type)
        compare = (f"        {rt} ra = {entry.name}({call}), rb = bcir_{entry.name}({call});\n"
                   f"        if (memcmp(&ra, &rb, sizeof ra) != 0) {{ printf(\"MISMATCH@%d\", "
                   f"trial); return 0; }}")
    elif entry.ret_type.is_float:
        # value compare, but a NaN result (e.g. asin out of domain) counts as equal to itself -- the
        # `x != x` legs catch NaN, which `==` alone would (correctly, per IEEE) report as unequal.
        rt = _cname(entry.ret_type)
        compare = (f"        {rt} ra = {entry.name}({call}), rb = bcir_{entry.name}({call});\n"
                   f"        if (!(ra == rb || (ra != ra && rb != rb))) {{ printf(\"MISMATCH@%d\", "
                   f"trial); return 0; }}")
    else:
        compare = (f"        if ({entry.name}({call}) != bcir_{entry.name}({call})) {{\n"
                   f"            printf(\"MISMATCH@%d\", trial); return 0; }}")
    return f"""#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <math.h>
{guard}{source}

{emitted}

{chr(10).join(prelude)}
static uint64_t _s = 0x9E3779B97F4A7C15u;
static uint32_t rng(void) {{ _s = _s * 6364136223846793005u + 1442695040888963407u; return (uint32_t)(_s >> 32); }}

int main(void) {{
{chr(10).join(decls)}
    for (int trial = 0; trial < 256; trial++) {{
{chr(10).join(setup)}
{compare}
    }}
    printf("MATCH");
    return 0;
}}
"""
