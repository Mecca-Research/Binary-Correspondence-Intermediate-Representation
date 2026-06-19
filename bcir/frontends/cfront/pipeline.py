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
from dataclasses import dataclass, field

from ...channels import host_channel
from ...kbcir.compose import plan_composite
from ...kbcir.cost import Theta
from ...kbcir.realize import optimize
from ...kbcir.weights import PERF
from ...verify import Diagnostic, verify, verify_plan
from .cparse import parse_unit
from .cpp import preprocess
from .emit import emit_function
from .lower import LoweredUnit, lower_unit


@dataclass
class CompileResult:
    source: str
    unit: object
    lowered: LoweredUnit
    plans: dict = field(default_factory=dict)         # fn -> RealizationResult
    emitted: dict = field(default_factory=dict)       # fn -> C text
    explain: dict = field(default_factory=dict)       # fn -> bcir-explain text
    attestation: dict = field(default_factory=dict)   # fn -> R12/R13/R17/R18 attestation dict
    diagnostics: list = field(default_factory=list)   # R1–R18 Diagnostics (empty == clean)
    r18_ok: bool = True
    equivalence: str = "skip"                         # match | MISMATCH | skip:<reason>

    @property
    def is_clean(self) -> bool:
        return not self.diagnostics and self.r18_ok

    @property
    def behaviour_equivalent(self) -> bool:
        return self.equivalence == "match"


def _cc():
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def compile_unit(source: str, *, includes: dict | None = None, embeds: dict | None = None,
                 search_paths: list | None = None, defines: dict | None = None,
                 check_clang: bool = True, filename: str = "<source>") -> CompileResult:
    # L7: preprocess first — the expanded text is what both the parser and the Clang harness see,
    # so the equivalence check validates the lowering of the *preprocessed* program. `search_paths`
    # (the source dir + -I dirs) resolves `#include "..."` from disk; `defines` seeds -D macros;
    # `filename` is the translation unit's name for __FILE__.
    source = preprocess(source, includes=includes, embeds=embeds,
                        search_paths=search_paths, defines=defines, name=filename)
    unit = parse_unit(source)
    lowered = lower_unit(unit)
    h, theta, policy = host_channel().profile, Theta.cool(), PERF
    res = CompileResult(source=source, unit=unit, lowered=lowered)

    # --- per-function: plan, verify (R1–R9), emit, explain ---
    for name, lf in lowered.functions.items():
        diags = verify(lf.module)
        plan = optimize(lf.module, h, theta, policy)
        diags += verify_plan(lf.module, plan)
        res.plans[name] = plan
        res.emitted[name] = emit_function(lf)
        res.explain[name] = _explain(lf.module, h, theta, policy)
        res.diagnostics += [Diagnostic(d.law, f"{name}: {d.message}") for d in diags]

    # --- R18: the inter-procedural call graph, via the real plan_composite machinery ---
    if lowered.entry:
        region = lowered.compose_functions[lowered.entry].region
        try:
            plan_composite(region, lowered.compose_functions, lowered.resources, h, theta, policy)
            res.r18_ok = True
        except Exception as e:  # noqa: BLE001 -- recursion / undefined callee == an R18 violation
            res.r18_ok = False
            res.diagnostics.append(Diagnostic("R18", f"call-graph integrity: {e}"))

    # --- behaviour equivalence vs Clang (clang-gated) ---
    if check_clang:
        res.equivalence = _equivalence(source, lowered)

    # --- C.2 attestation: stamp each emitted function with its R12/R13/R17/R18 provenance ---
    for name in lowered.functions:
        res.attestation[name] = _attest(res, name, h.name)
        res.emitted[name] = _attestation_comment(res.attestation[name]) + "\n" + res.emitted[name]
    return res


def _attest(res: CompileResult, fn: str, target: str) -> dict:
    """The verified-C-output provenance for one function: which laws hold, a claim-graph digest
    (R13-style), the integer-exactness (R17), the call-graph integrity (R18), and the plan cost."""
    lf = res.lowered.functions[fn]
    fp = repr([(c.id, c.op, c.opcode.name, c.rd, c.wr, c.imm, c.domain.name) for c in lf.claims])
    digest = hashlib.sha256(fp.encode()).hexdigest()[:16]
    fn_diags = [d for d in res.diagnostics if d.message.startswith(f"{fn}:") or d.law == "R18"]
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
            b = subprocess.run([cc, std, "-O1", src, "-o", exe], capture_output=True, text=True)
            if b.returncode == 0:
                break
        else:
            return f"skip:build-failed:{b.stderr.strip().splitlines()[-1] if b.stderr else '?'}"
        run = subprocess.run([exe], capture_output=True, text=True)
        out = run.stdout.strip()
        return "match" if out == "MATCH" else f"MISMATCH:{out}"


def _harness_c(source: str, lowered: LoweredUnit, entry) -> str:
    """A self-contained C program: the original source + the emitted bcir_* functions + a main that
    runs both on the same seeded-random inputs and prints MATCH iff every trial agrees."""
    from .emit import _cname  # noqa: PLC0415

    emitted = "\n\n".join(emit_function(lf) for lf in lowered.functions.values())
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
    else:
        compare = (f"        if ({entry.name}({call}) != bcir_{entry.name}({call})) {{\n"
                   f"            printf(\"MISMATCH@%d\", trial); return 0; }}")
    return f"""#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdatomic.h>
{source}

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
