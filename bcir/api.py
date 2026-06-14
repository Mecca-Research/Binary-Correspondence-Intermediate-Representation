"""bcir.api -- the small, stable embeddable facade (library-first).

One call turns a program into a deployable kernel artifact: a portable C23 kernel
plus a freestanding header (the ABI a resident toolchain compiles against),
metadata (the K_BCIR-selected lane geometry, score, calibration generation), the
provenance manifest digest (the plan's commit hash), and an **R12 attestation**
(the C kernel was checked to preserve the selected realization). The same
artifact serves both deployment shapes:

  * **AOT** -- `build_artifact(...).kernel_c` compiled by the host toolchain
    (`compile_kernel(..., run=True)` self-checks it here);
  * **driver-embedded** -- emit `kernel_c` + `header_c`, hand them to the
    resident compiler at execution time (the StreamPack-rehydration story).

This is the public surface a driver/runtime calls; the rest of `bcir.*` is the
implementation behind it. Pure and deterministic: no compilation happens unless
you ask (`run=True`).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

from .examples import PROGRAMS
from .kbcir import TARGETS, build_manifest, optimize
from .kbcir.cost import Theta
from .kbcir.weights import PERF, POLICIES
from .lower.c_kernel import C_OP, compile_and_run_c, emit_header_c, emit_kernel_c
from .lower.llvm import find_elementwise
from .verify import verify_c_lowering

_THETAS = {"cool": Theta.cool(), "hot": Theta.hot(), "mem_bound": Theta.mem_bound()}


@dataclass(frozen=True)
class KernelArtifact:
    """A deployable kernel: C source + ABI header + metadata + R12 attestation."""

    fn_name: str
    program: str
    target: str
    elem: str
    op: str
    width: int
    score: int
    cal_gen: int
    manifest_digest: int
    attested: bool                 # R12 (verify_c_lowering) passed with no diagnostics
    diagnostics: tuple             # ((law, message), ...) -- empty iff attested
    kernel_c: str
    header_c: str

    def to_files(self, dirpath: str) -> dict:
        """Write `<fn>.c` + `<fn>.h` for a resident toolchain; returns the paths."""
        os.makedirs(dirpath, exist_ok=True)
        cpath = os.path.join(dirpath, f"{self.fn_name}.c")
        hpath = os.path.join(dirpath, f"{self.fn_name}.h")
        with open(cpath, "w", encoding="utf-8") as f:
            f.write(self.kernel_c)
        with open(hpath, "w", encoding="utf-8") as f:
            f.write(self.header_c)
        return {"kernel": cpath, "header": hpath}

    def metadata(self) -> dict:
        d = asdict(self)
        for k in ("kernel_c", "header_c", "diagnostics"):
            d.pop(k, None)
        d["diagnostics"] = [list(x) for x in self.diagnostics]
        return d

    def metadata_json(self) -> str:
        return json.dumps(self.metadata(), indent=2, sort_keys=True)


def _resolve(program):
    """A program name (examples.PROGRAMS) or an already-built Module."""
    if isinstance(program, str):
        if program not in PROGRAMS:
            raise KeyError(f"unknown program {program!r}; known: {sorted(PROGRAMS)}")
        return program, PROGRAMS[program]()
    return getattr(program, "name", "module"), program


def build_artifact(program, *, target: str = "x86_avx512", theta: str = "cool",
                   policy: str = "latency", elem: str = "f32", table=None,
                   fn_name: str = "bcir_kernel") -> KernelArtifact:
    """Plan a program for a target/Θ and emit a deployable, R12-attested kernel
    artifact. `table` is an optional frozen `CalibratedProfile` (applies measured
    constants). Pure -- no compilation."""
    name, module = _resolve(program)
    h = TARGETS[target]
    if table is not None:
        h = table.apply(h)
    th = _THETAS.get(theta, Theta.cool())
    pol = POLICIES.get(policy, PERF)

    result = optimize(module, h, th, pol)
    claim, cand = find_elementwise(module, result)  # raises if not lowerable
    kernel_c = emit_kernel_c(module, result, fn_name, elem)
    header_c = emit_header_c(fn_name, elem)
    diags = verify_c_lowering(module, result, kernel_c, elem)
    manifest = build_manifest(module, h, th, pol)

    return KernelArtifact(
        fn_name=fn_name, program=name, target=h.name, elem=elem,
        op=C_OP.get(claim.opcode, "?"), width=int(cand.width), score=int(result.score),
        cal_gen=int(getattr(h, "cal_gen", 0)), manifest_digest=int(manifest.digest),
        attested=not diags, diagnostics=tuple((d.law, d.message) for d in diags),
        kernel_c=kernel_c, header_c=header_c)


def compile_kernel(program, *, run: bool = False, **kw):
    """Build the artifact; if `run`, also compile + self-check it via the host
    toolchain (AOT). Returns the artifact, or (artifact, (ok, output)) when run."""
    art = build_artifact(program, **kw)
    if not run:
        return art
    _, module = _resolve(program)
    h = TARGETS[kw.get("target", "x86_avx512")]
    if kw.get("table") is not None:
        h = kw["table"].apply(h)
    th = _THETAS.get(kw.get("theta", "cool"), Theta.cool())
    pol = POLICIES.get(kw.get("policy", "latency"), PERF)
    result = optimize(module, h, th, pol)        # deterministic: same plan as build_artifact
    ok, out = compile_and_run_c(module, result, fn_name=art.fn_name, elem=kw.get("elem", "f32"))
    return art, (ok, out)
