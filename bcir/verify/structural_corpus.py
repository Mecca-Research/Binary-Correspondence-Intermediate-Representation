"""The shared structural-law corpus (S0-6): one corpus, two runners, every mismatch a finding.

Rows 9, 12, 13, 17, 19, 20 and 21 of the 2026-07/08 assessment each named a structural law the
two rails enforced differently, or one rail not at all: widths, alignments, shapes and strides
(the oracle admitted what the law rail's op verifiers refused at parse), phase identity and
ordering (dangling and duplicate phase ids; five phase orders in use), the address width of a
first-class MMIO/atomic access (i32 accepted under a 64-bit target, then `inttoptr`'d), the
domain contract MAP/ROP derived for non-RAM resources (every claim defaulted to RAM), the M5
descriptors (nothing validated them at construction), the R13 manifest record (arity, artifact
order and certified constants unchecked on the law rail) and the convolution whose one-tile
realization overflowed signed arithmetic in the GEM lowerer.

This module is the ONE corpus those laws are now held to on BOTH rails:

  * `CASES` -- every case is a rail-neutral spec plus the verdict each rail must reach: the law
    family, the oracle's diagnostic substring, the law rail's diagnostic substring;
  * the ORACLE runner (`run_oracle`) builds the Python artifact -- a `Module`, a `TargetProfile`,
    an M5 descriptor, a `ProvenanceManifest`, a `ConvSpec`, a parsed MAP/ROP program -- and reads
    the verdict off `bcir.verify` or off the constructor;
  * the LAW-RAIL runner renders the same spec as BCIR-MLIR (`render`) and drives `bcir-opt` over
    it (`run_mlir`), and `emit_fixture` projects every law-rail case into ONE `-verify-diagnostics`
    fixture (`mlir/test/passes/structural_corpus.mlir`, generated, drift-gated with `--check`)
    that `tools/wsl/check_passes.sh` executes on every MLIR job;
  * `findings` compares each rail's verdict with the corpus: a legal case refused, an illegal
    case admitted, a refusal for another reason or under another law -- every one is a finding,
    never a pass (laws.md L2/L11).

    python -m bcir.verify.structural_corpus --emit     # regenerate the fixture
    python -m bcir.verify.structural_corpus --check    # refuse drift (the quick tier runs this)
    python -m bcir.verify.structural_corpus --run [--bcir-opt PATH]   # both rails, findings

A case names the rails it runs on. A law with no oracle-side artifact (a symbol reference that
cannot dangle in a Python object graph) or no law-rail one (a descriptor the dialect has no op
for) is declared as such in the case, never inferred: every case is a positive statement of
which rail enforces what.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from ..model import (
    ISOLATED_DOMAINS,
    Claim,
    Domain,
    Lane,
    Module,
    Opcode,
    Phase,
    Resource,
    StrideClass,
)

ORACLE = "oracle"
MLIR = "mlir"
RAILS = (ORACLE, MLIR)
FIXTURE = Path("mlir/test/passes/structural_corpus.mlir")
_ROOT = Path(__file__).resolve().parents[2]
_ERR = "\x00ERR\x00"  # renderer sentinel: the line that carries the law-rail diagnostic

_DOMAIN_SPELL = {
    Domain.RAM: "ram",
    Domain.VRAM: "vram",
    Domain.NVM: "nvm",
    Domain.MMIO: "mmio",
    Domain.CXL: "cxl",
    Domain.HBM: "hbm",
}
_LANE_SPELL = {Lane.U: "u", Lane.UX: "ux", Lane.T: "t", Lane.GGG: "ggg", Lane.A: "a", Lane.H: "h"}
_STRIDE_SPELL = {
    StrideClass.SCALAR: "scalar",
    StrideClass.UNIT: "unit",
    StrideClass.STRIDED: "strided",
    StrideClass.CACHELINE: "cacheline",
    StrideClass.TILE: "tile",
    StrideClass.RANDOM: "random",
}
_DOMAINS = {v: k for k, v in _DOMAIN_SPELL.items()}
_LANES = {v: k for k, v in _LANE_SPELL.items()}
_STRIDES = {v: k for k, v in _STRIDE_SPELL.items()}
# The four hashes of the canonical vector_add manifest (verify_provenance.mlir), so the R13
# cases fold real component values through the same FNV chain on both rails.
_MM, _MT, _MTH, _MP = (
    7127522701151166272,
    5864064355688965777,
    1870846051561339781,
    4048695575545564183,
)
_LATENCY_WEIGHTS = (2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1)


@dataclass(frozen=True)
class Case:
    """One corpus entry: a rail-neutral spec and the verdict every rail it runs on must reach."""

    name: str
    kind: str  # module | map | rop | target | mmio | manifest | portfolio | calibration
    #          | binary | stream | fsm | grammar | conv
    spec: dict
    law: str = ""  # the law family ("" = legal on every rail the case runs on)
    oracle: str = ""  # expected oracle diagnostic substring (an illegal case)
    mlir: str = ""  # expected law-rail diagnostic substring (an illegal case)
    rails: tuple[str, ...] = RAILS
    note: str = ""
    also_laws: tuple[str, ...] = ()  # further law families BOTH rails report on this case
    mlir_also: tuple[str, ...] = ()  # the law rail's further diagnostics (one expectation each)

    @property
    def legal(self) -> bool:
        return not self.law

    @property
    def mlir_expected(self) -> tuple[str, ...]:
        return ((self.mlir,) if self.mlir else ()) + self.mlir_also


@dataclass(frozen=True)
class Verdict:
    rail: str
    refused: bool
    messages: tuple[str, ...] = ()
    laws: tuple[str, ...] = ()  # the R-law families the rail named ("" entries for op verifiers)


@dataclass(frozen=True)
class Finding:
    case: str
    rail: str
    kind: str  # refused-legal | admitted-illegal | message | law | no-verdict
    detail: str


# --- the corpus --------------------------------------------------------------------------------


def _module(resources=(), phases=(), claims=(), name="m") -> dict:
    return {
        "name": name,
        "resources": list(resources),
        "phases": list(phases),
        "claims": list(claims),
    }


def _res(rid, shape=(8,), domain="ram", align=64, elem_bytes=4, access="flat", err=False) -> dict:
    return {
        "rid": rid,
        "shape": list(shape),
        "domain": domain,
        "align": align,
        "elem_bytes": elem_bytes,
        "access": access,
        "err": err,
    }


def _phase(pid, deps=(), sym=None, err=False) -> dict:
    return {"id": pid, "deps": list(deps), "sym": sym or f"p{pid}", "err": err}


def _claim(cid=1, phase=0, rd=(1,), wr=(1,), count=8, **kw) -> dict:
    c = {
        "id": cid,
        "phase": phase,
        "phase_sym": kw.pop("phase_sym", None) or f"p{phase}",
        "op": "vector.add",
        "opcode": "ADD",
        "rd": list(rd),
        "wr": list(wr),
        "count": count,
        "lane": "u",
        "stride_class": "unit",
        "stride_k": 1,
        "offset": 0,
        "domain": "ram",
        "hazard": "unique",
        "verify": "bounds",
        "bounds": "strict",
        "volatile": False,
        "err": False,
    }
    c.update(kw)
    return c


def _conv(in_c, in_h, in_w, out_c, kh, kw, stride=1, pad=0, dtype="f32") -> dict:
    return {
        "in_c": in_c,
        "in_h": in_h,
        "in_w": in_w,
        "out_c": out_c,
        "kh": kh,
        "kw": kw,
        "stride": stride,
        "pad": pad,
        "dtype": dtype,
    }


def _mmio(triple, addr_bits, op="load") -> dict:
    return {"triple": triple, "addr_bits": addr_bits, "op": op}


_COMPONENTS = {"m_module": _MM, "m_target": _MT, "m_theta": _MTH, "m_policy": _MP}


def _manifest(
    artifacts=(), *, names=None, gens=None, components=None, with_theta=False, with_claims=False
) -> dict:
    """A manifest record: the declared component hashes (all four unless `components` names a
    subset) and the digest folded over exactly those (0 for an undeclared one) -- the FNV chain
    provenance._digest and the law rail's recompute both walk."""
    from ..kbcir.provenance import _digest

    arts = tuple(artifacts)
    declared = dict(_COMPONENTS) if components is None else {k: _COMPONENTS[k] for k in components}
    folded = [declared.get(k, 0) for k in ("m_module", "m_target", "m_theta", "m_policy")]
    return {
        "digest": _digest(*folded, arts),
        "score": 7808,
        "artifacts": [list(a) for a in arts],
        "names": names,  # law rail only: an explicit (possibly mismatched) name array
        "gens": gens,
        "components": declared,
        "with_theta": with_theta,
        "with_claims": with_claims,
    }


def _binary(fields, size_bits=0, endianness="little", alignment_bits=8) -> dict:
    return {
        "fields": [list(f) for f in fields],  # (name, offset, width, kind)
        "record_fields": None,  # law rail only: explicit field refs (a dangling one)
        "size_bits": size_bits,
        "endianness": endianness,
        "alignment_bits": alignment_bits,
    }


def _fsm(states, transitions, start="s0") -> dict:
    return {
        "states": [list(s) for s in states],  # (name, accepting, error)
        "transitions": [list(t) for t in transitions],  # (src, dst, on)
        "start": start,
    }


_MAP_HBM = "res A rid 10 n 8 domain hbm\nres C rid 12 n 8 domain hbm\nadd C <- A, A n 8\n"
_MAP_MMIO_W = "res A rid 10 n 8\nres R rid 5 n 1 domain mmio\nadd R <- A n 1 hazard barriered\n"
_MAP_MMIO_W_UNIQUE = "res A rid 10 n 8\nres R rid 5 n 1 domain mmio\nadd R <- A n 1\n"
_ROP_HBM = (
    "module m { resource A { rid 10 domain hbm count 8 } resource C { rid 12 domain hbm count 8 }"
    " phase 0 { claim add { op add reads A writes C count 8 } } }"
)
_ROP_MMIO_R = (
    "module m { resource R { rid 5 domain mmio count 1 } resource C { rid 12 count 8 }"
    " phase 0 { claim rd { op load reads R writes C count 1 hazard barriered } } }"
)
_ROP_MMIO_R_UNIQUE = (
    "module m { resource R { rid 5 domain mmio count 1 } resource C { rid 12 count 8 }"
    " phase 0 { claim rd { op load reads R writes C count 1 } } }"
)

CASES: tuple[Case, ...] = (
    # ---- registry well-formedness (R1 on the oracle; the bcir.resource op verifier) ----------
    Case("resource.legal", "module", _module([_res(1)], [_phase(0)], [_claim()])),
    Case(
        "resource.shape.unknown",
        "module",
        _module([_res(1, shape=())], [_phase(0)], [_claim()]),
        note="an empty shape is an UNKNOWN extent on both rails: R7 cannot check it, R1 admits it",
    ),
    Case(
        "resource.shape.zero",
        "module",
        _module([_res(1), _res(2, shape=(0,), err=True)], [_phase(0)], [_claim()]),
        "R1",
        "resource 2: shape extents must be positive (got 0)",
        "shape extents must be positive (got 0)",
    ),
    Case(
        "resource.shape.negative",
        "module",
        _module([_res(1), _res(2, shape=(-4,), err=True)], [_phase(0)], [_claim()]),
        "R1",
        "resource 2: shape extents must be positive (got -4)",
        "shape extents must be positive (got -4)",
    ),
    Case(
        "resource.shape.overflow",
        "module",
        _module([_res(1), _res(2, shape=(1 << 32, 1 << 32), err=True)], [_phase(0)], [_claim()]),
        "R1",
        "resource 2: shape element count exceeds signed 64-bit range",
        "shape element count exceeds signed 64-bit range",
    ),
    Case(
        "resource.align.not_pow2",
        "module",
        _module([_res(1, align=3, err=True)], [_phase(0)], [_claim()]),
        "R1",
        "resource 1: align must be a positive power of two (got 3)",
        "align must be a positive power of two (got 3)",
    ),
    Case(
        "resource.align.zero",
        "module",
        _module([_res(1, align=0, err=True)], [_phase(0)], [_claim()]),
        "R1",
        "resource 1: align must be a positive power of two (got 0)",
        "align must be a positive power of two (got 0)",
    ),
    Case(
        "resource.elem_bytes.zero",
        "module",
        _module([_res(1, elem_bytes=0)], [_phase(0)], [_claim()]),
        rails=(ORACLE,),
        note="legal: the dialect's resource carries no element width (it rides "
        "target.capability) and cfront's zero-size objects declare 0 -- not a law on either rail",
    ),
    # ---- access-pattern well-formedness (R7 on the oracle; the bcir.claim op verifier) --------
    Case("claim.count.zero", "module", _module([_res(1)], [_phase(0)], [_claim(count=0)])),
    Case(
        "claim.count.negative",
        "module",
        _module([_res(1)], [_phase(0)], [_claim(count=-1, err=True)]),
        "R7",
        "claim 1: count must be non-negative (got -1)",
        "count must be non-negative (got -1)",
    ),
    Case(
        "claim.offset.negative",
        "module",
        _module([_res(1)], [_phase(0)], [_claim(offset=-1, err=True)]),
        "R7",
        "claim 1: offset must be non-negative (got -1)",
        "offset must be non-negative (got -1)",
    ),
    Case(
        "claim.stride_k.zero",
        "module",
        _module([_res(1)], [_phase(0)], [_claim(stride_class="strided", stride_k=0, err=True)]),
        "R7",
        "claim 1: stride_k must be positive (got 0)",
        "stride_k must be positive (got 0)",
    ),
    Case(
        "claim.stride_k.negative",
        "module",
        _module([_res(1)], [_phase(0)], [_claim(stride_class="strided", stride_k=-2, err=True)]),
        "R7",
        "claim 1: stride_k must be positive (got -2)",
        "stride_k must be positive (got -2)",
    ),
    Case(
        "claim.strided.legal",
        "module",
        _module([_res(1)], [_phase(0)], [_claim(stride_class="strided", stride_k=4, count=2)]),
        note="extent 0 + (2-1)*4 + 1 = 5 <= 8",
    ),
    Case(
        "claim.strided.overrun",
        "module",
        _module(
            [_res(1)], [_phase(0)], [_claim(stride_class="strided", stride_k=4, count=3, err=True)]
        ),
        "R7",
        "claim 1: read of RID 1 overruns the resource (extent 9 > 8)",
        "R7: claim c1 read of @r1 overruns the resource (extent 9 > 8)",
    ),
    Case(
        "claim.extent.overflow",
        "module",
        _module(
            [_res(1)],
            [_phase(0)],
            [_claim(stride_class="strided", stride_k=4, count=1 << 62, err=True)],
        ),
        "R7",
        "claim 1: affine access extent exceeds signed 64-bit range",
        "R7: affine access extent exceeds signed 64-bit range",
    ),
    Case(
        "claim.undeclared_resource",
        "module",
        _module([_res(1)], [_phase(0)], [_claim(rd=(1, 999), err=True)]),
        "R2",
        "claim 1 references undeclared RID 999",
        "R2: claim c1 reads undeclared resource @r999",
    ),
    Case(
        "lane.tile_on_unit",
        "module",
        _module([_res(1)], [_phase(0)], [_claim(lane="t", err=True)]),
        "R6",
        "claim 1: lane T illegal for stride_class UNIT",
        "R6: claim c1 lane illegal for its stride class",
    ),
    # ---- phase identity and ordering (R4) ------------------------------------------------------
    Case(
        "phase.duplicate_id",
        "module",
        _module([_res(1)], [_phase(0), _phase(0, sym="q0", err=True)], [_claim()]),
        "R4",
        "duplicate phase id 0",
        "R4: phase @q0 duplicates phase id 0 of @p0",
    ),
    Case(
        "phase.dangling_dep",
        "module",
        _module([_res(1)], [_phase(0, deps=(9,), err=True)], [_claim()]),
        "R4",
        "phase 0 depends on undeclared phase 9",
        "R4: phase @p0 depends on undeclared phase @p9",
    ),
    Case(
        "phase.self_dep",
        "module",
        _module([_res(1)], [_phase(0, deps=(0,), err=True)], [_claim()]),
        "R4",
        "phase dependency graph contains a cycle",
        "R4: phase dependency cycle through @p0",
    ),
    Case(
        "phase.cycle",
        "module",
        _module([_res(1)], [_phase(0, deps=(1,), err=True), _phase(1, deps=(0,))], [_claim()]),
        "R4",
        "phase dependency graph contains a cycle",
        "R4: phase dependency cycle through @p0",
    ),
    Case(
        "phase.order.ids_out_of_dependency_order",
        "module",
        _module(
            [_res(1)],
            [_phase(1), _phase(0, deps=(1,))],
            [_claim(cid=1, phase=0), _claim(cid=2, phase=1)],
        ),
        note="legal; the canonical order is [1, 0] on both rails (schedule_phase_order.mlir "
        "pins -bcir-schedule's exec_order to it)",
    ),
    # ---- domain legality (R3): the isolated-domain rule, one rule on both rails ----------------
    Case(
        "domain.hbm_only",
        "module",
        _module([_res(1, domain="hbm")], [_phase(0)], [_claim(domain="hbm")]),
    ),
    Case(
        "domain.host_mix",
        "module",
        _module([_res(1), _res(2, domain="hbm")], [_phase(0)], [_claim(rd=(1,), wr=(2,))]),
        note="RAM/HBM are mutually addressable host memory: a RAM claim may accumulate into HBM",
    ),
    Case(
        "domain.unbacked",
        "module",
        _module([_res(1, domain="hbm")], [_phase(0)], [_claim(err=True)]),
        "R3",
        "claim 1: declares domain RAM but touches only {HBM}",
        "R3: claim c1 declares a domain not backed by any touched resource",
    ),
    Case(
        "domain.ram_claim_reaches_mmio",
        "module",
        _module(
            [_res(1), _res(2, domain="mmio", shape=(1,))],
            [_phase(0)],
            [_claim(rd=(1, 2), count=1, err=True)],
        ),
        "R3",
        "claim 1: read of RID 2 (domain MMIO) does not match the claim domain RAM",
        "R3: claim c1 reads @r2 (domain mmio) does not match the claim domain ram -- the resource "
        "is in a device-isolated domain",
    ),
    Case(
        "domain.mmio_claim_with_ram_operand",
        "module",
        _module(
            [_res(1), _res(2, domain="mmio", shape=(1,))],
            [_phase(0)],
            [_claim(rd=(1,), wr=(2,), count=1, domain="mmio", hazard="barriered", volatile=True)],
        ),
        note="the shape of every cfront MMIO write: the RAM value stored into the register",
    ),
    Case(
        "domain.mmio_write_unique_hazard",
        "module",
        _module(
            [_res(2, domain="mmio", shape=(1,))],
            [_phase(0)],
            [_claim(rd=(2,), wr=(2,), count=1, domain="mmio", err=True)],
        ),
        "R3",
        "claim 1: MMIO write to RID 2 requires an atomic/barriered hazard contract",
        "R3: claim c1 MMIO write to @r2 requires an atomic/barriered hazard",
    ),
    Case(
        "domain.nvm_stage_to_vram",
        "module",
        _module(
            [_res(1, domain="nvm"), _res(2, domain="vram")],
            [_phase(0)],
            [_claim(rd=(1,), wr=(2,), domain="vram")],
        ),
        note="NVM is a memory TIER the HAM fabric stages across, not a device-isolated domain: "
        "the law rail's former 'NVM cell' isolation had no fixture, and the oracle's tier model won",
    ),
    Case(
        "domain.mmio_claim_nvm_operand",
        "module",
        _module(
            [_res(2, domain="mmio", shape=(1,)), _res(3, domain="nvm", shape=(1,))],
            [_phase(0)],
            [_claim(rd=(3,), wr=(2,), count=1, domain="mmio", hazard="barriered", volatile=True)],
        ),
        note="a device register written from a tier operand: only the resource side is isolated",
    ),
    Case(
        "domain.ham_on_mmio",
        "module",
        _module([_res(2, domain="mmio", shape=(1,), access="ham", err=True)], [_phase(0)], []),
        "R3",
        "resource 2: HAM access is illegal in the MMIO domain",
        "R3: resource r2 HAM access is illegal in the MMIO domain",
    ),
    # ---- the target descriptor (TargetProfile construction; the target.capability verifier) --
    Case("target.legal", "target", {"cacheline": 64, "lane_widths": [1, 8, 16], "elem_bytes": 4}),
    Case(
        "target.cacheline.not_pow2",
        "target",
        {"cacheline": 48, "lane_widths": [1, 8], "elem_bytes": 4},
        "R8",
        "cacheline must be a positive power of two (got 48)",
        "cacheline must be a positive power of two (got 48)",
        note="an op-level check on the law rail; the oracle refuses at construction -- filed under "
        "the cost law the descriptor feeds",
    ),
    Case(
        "target.lane_widths.empty",
        "target",
        {"cacheline": 64, "lane_widths": [], "elem_bytes": 4},
        "R8",
        "lane_widths must contain at least one positive width",
        "lane_widths must contain at least one positive width",
    ),
    Case(
        "target.lane_width.zero",
        "target",
        {"cacheline": 64, "lane_widths": [1, 0], "elem_bytes": 4},
        "R8",
        "lane widths must be positive (got 0)",
        "lane widths must be positive (got 0)",
    ),
    Case(
        "target.elem_bytes.zero",
        "target",
        {"cacheline": 64, "lane_widths": [1, 8], "elem_bytes": 0},
        "R8",
        "elem_bytes must be positive (got 0)",
        "affinity_domains, mem_channels, mem_unit, and elem_bytes must be positive",
    ),
    # ---- address width (R12): the triple -> pointer-width table, mirrored ---------------------
    Case("addr.i64_under_x86_64", "mmio", _mmio("x86_64-avx512", 64)),
    Case(
        "addr.i32_under_x86_64",
        "mmio",
        _mmio("x86_64-avx512", 32),
        "R12",
        "the device-register address is 32 bits but the target 'x86_64-avx512' addresses 64-bit "
        "pointers; the inttoptr lowering would leave it zero-extended",
        "R12: the device-register address is 32 bits but the target 'x86_64-avx512' addresses "
        "64-bit pointers; the inttoptr lowering would leave it zero-extended",
    ),
    Case(
        "addr.i32_store_under_aarch64",
        "mmio",
        _mmio("aarch64-sve", 32, op="store"),
        "R12",
        "the device-register address is 32 bits but the target 'aarch64-sve' addresses 64-bit",
        "R12: the device-register address is 32 bits but the target 'aarch64-sve' addresses 64-bit",
    ),
    Case("addr.i32_under_riscv32", "mmio", _mmio("riscv32-unknown-elf", 32)),
    Case(
        "addr.i64_under_riscv32",
        "mmio",
        _mmio("riscv32-unknown-elf", 64),
        "R12",
        "the device-register address is 64 bits but the target 'riscv32-unknown-elf' addresses "
        "32-bit pointers; the inttoptr lowering would leave it truncated",
        "R12: the device-register address is 64 bits but the target 'riscv32-unknown-elf' "
        "addresses 32-bit pointers; the inttoptr lowering would leave it truncated",
    ),
    Case(
        "addr.rmw_i32_under_nvptx64",
        "mmio",
        _mmio("nvptx64-warp", 32, op="rmw"),
        "R12",
        "the object address is 32 bits but the target 'nvptx64-warp' addresses 64-bit pointers",
        "R12: the object address is 32 bits but the target 'nvptx64-warp' addresses 64-bit pointers",
    ),
    Case(
        "addr.cas_i32_under_bpf",
        "mmio",
        _mmio("bpfel", 32, op="cas"),
        "R12",
        "the object address is 32 bits but the target 'bpfel' addresses 64-bit pointers",
        "R12: the object address is 32 bits but the target 'bpfel' addresses 64-bit pointers",
    ),
    Case("addr.cas_i64_under_bpf", "mmio", _mmio("bpfel", 64, op="cas")),
    Case(
        "addr.i32_unknown_triple",
        "mmio",
        _mmio("x", 32),
        note="an architecture the table does not know: the law is vacuous, the >= 32-bit floor holds",
    ),
    Case("addr.i32_no_target", "mmio", _mmio(None, 32), note="no target in scope: only the floor"),
    # ---- the R13 manifest record ---------------------------------------------------------------
    Case("manifest.legal", "manifest", _manifest((("cal_gen", 4), ("map_gen", 2)))),
    Case("manifest.legal.no_artifacts", "manifest", _manifest(())),
    Case(
        "manifest.unsorted_artifacts",
        "manifest",
        _manifest((("map_gen", 2), ("cal_gen", 4))),
        "R13",
        "provenance artifacts must be sorted with unique names",
        "R13: manifest artifact names must be sorted and unique ('map_gen' precedes 'cal_gen')",
        note="the digest was folded in the declared order, which the old law rail accepted",
    ),
    Case(
        "manifest.duplicate_artifact",
        "manifest",
        _manifest((("cal_gen", 4), ("cal_gen", 2))),
        "R13",
        "provenance artifacts must be sorted with unique names",
        "R13: manifest artifact names must be sorted and unique ('cal_gen' precedes 'cal_gen')",
    ),
    Case(
        "manifest.arity_mismatch",
        "manifest",
        _manifest((("cal_gen", 4), ("map_gen", 2)), names=["cal_gen"], gens=[4, 2]),
        "R13",
        mlir="R13: manifest artifact_names/artifact_gens arity mismatch (1 names, 2 generations)",
        rails=(MLIR,),
        note="a (name, generation) pair cannot mismatch its own arity in the oracle's object model",
    ),
    Case(
        "manifest.absent_theta",
        "manifest",
        _manifest((), components=("m_theta",), with_claims=True),
        "R13",
        mlir="R13: manifest declares m_theta but the module carries no kbcir.theta to recompute it from",
        rails=(MLIR,),
        note="a manifest inside a module carrying a claim graph declares m_theta with no theta op; "
        "the oracle's verify_manifest recomputes from the inputs it is handed and cannot be asked "
        "about an object that is not there",
    ),
    Case(
        "manifest.theta_present",
        "manifest",
        _manifest((), components=("m_theta",), with_theta=True),
        note="the cool theta's hash, recomputed from the kbcir.theta op it sits beside",
    ),
    # ---- the R13 portfolio and calibration records ---------------------------------------------
    Case("portfolio.certified_gen1", "portfolio", {"entries": [["latency", 1, True]], "certs": []}),
    Case(
        "portfolio.certified_gen0",
        "portfolio",
        {"entries": [["latency", 0, True]], "certs": []},
        "R13",
        "policy 'latency' has no generation tag",
        "R13: portfolio generation tags must be >= 1",
    ),
    Case(
        "portfolio.promoted_without_certificate",
        "portfolio",
        {"entries": [["latency", 2, True]], "certs": []},
        "R13",
        "promoted policy 'latency' (gen 2) has no admitting replay certificate",
        "R13: promoted policy @latency (gen 2) has no admitting replay certificate",
    ),
    Case(
        "portfolio.promoted_uncertified_without_certificate",
        "portfolio",
        {"entries": [["latency", 2, False]], "certs": []},
        "R13",
        "promoted policy 'latency' (gen 2) has no admitting replay certificate",
        "R13: promoted policy @latency (gen 2) has no admitting replay certificate",
        note="the oracle used to skip uncertified entries; a promotion is witnessed or it did not happen",
    ),
    Case(
        "portfolio.promoted_with_certificate",
        "portfolio",
        {"entries": [["latency", 2, True]], "certs": [["latency", "latency", 4, 0]]},
    ),
    Case(
        "calibration.constants_match",
        "calibration",
        {
            "cal_gen": 1,
            "gather_penalty": 32,
            "base_overhead": 4,
            "random_q8": 8192,
            "strided_q8": 256,
        },
    ),
    Case(
        "calibration.constants_drift",
        "calibration",
        {
            "cal_gen": 1,
            "gather_penalty": 32,
            "base_overhead": 4,
            "random_q8": 16384,
            "strided_q8": 256,
        },
        "R13",
        "profile constants drifted from the certified table (cal_gen 1)",
        "R13: capability @cpu constants drifted from calibration certificate @cal (gather_penalty "
        "32 vs certified 64, base_overhead 4 vs certified 4, mem_unit 1 vs 1)",
    ),
    # ---- the M5 descriptors (construction on the oracle; op verifiers on the law rail) --------
    Case(
        "m5.binary.legal",
        "binary",
        _binary(
            [("opcode", 0, 8, "u"), ("command_id", 16, 16, "u"), ("namespace_id", 32, 32, "u")],
            size_bits=64,
        ),
        note="the NVMe SQE header",
    ),
    Case(
        "m5.field.width_zero",
        "binary",
        _binary([("a", 0, 0, "u")]),
        "R21",
        "field 'a': width_bits must be positive (got 0)",
        "field 'a': width_bits must be positive (got 0)",
    ),
    Case(
        "m5.field.offset_negative",
        "binary",
        _binary([("a", -8, 8, "u")]),
        "R21",
        "field 'a': offset_bits must be non-negative (got -8)",
        "field 'a': offset_bits must be non-negative (got -8)",
    ),
    Case(
        "m5.field.kind_unknown",
        "binary",
        _binary([("a", 0, 8, "q")]),
        "R21",
        "field 'a': kind must be one of u|s|f|bytes (got 'q')",
        "field 'a': kind must be one of u|s|f|bytes (got 'q')",
    ),
    Case(
        "m5.record.overlap",
        "binary",
        _binary([("b", 0, 32, "u"), ("c", 16, 32, "u")]),
        "R21",
        "record 'r': fields 'b' and 'c' overlap (bits [0, 32) and [16, 48))",
        "record 'r': fields 'b' and 'c' overlap (bits [0, 32) and [16, 48))",
    ),
    Case(
        "m5.record.exceeds_size",
        "binary",
        _binary([("b", 0, 32, "u")], size_bits=16),
        "R21",
        "record 'r': field 'b' ends at bit 32, beyond the record's 16 bits",
        "record 'r': field 'b' ends at bit 32, beyond the record's 16 bits",
    ),
    Case(
        "m5.record.dangling_field",
        "binary",
        _binary([("b", 0, 32, "u")]) | {"record_fields": ["b", "ghost"]},
        "R21",
        mlir="record 'r': field @ghost does not resolve to a bcir.binary.field",
        rails=(MLIR,),
        note="a record's fields are objects on the oracle; only a symbol reference can dangle",
    ),
    Case(
        "m5.format.endianness_unknown",
        "binary",
        _binary([("b", 0, 32, "u")], endianness="middle"),
        "R21",
        "format 'f': endianness must be one of little|big|le|be (got 'middle')",
        "format 'f': endianness must be one of little|big|le|be (got 'middle')",
    ),
    Case(
        "m5.format.alignment_zero",
        "binary",
        _binary([("b", 0, 32, "u")], alignment_bits=0),
        "R21",
        "format 'f': alignment_bits must be positive (got 0)",
        "format 'f': alignment_bits must be positive (got 0)",
    ),
    Case(
        "m5.stream.legal",
        "stream",
        {"kind": "binary", "encoding": "le", "element_bits": 8, "max_window": 64},
    ),
    Case(
        "m5.stream.kind_unknown",
        "stream",
        {"kind": "smoke", "encoding": "le", "element_bits": 8, "max_window": 64},
        "R21",
        "stream 's': kind must be one of text|binary|telemetry|packet|driver|token (got 'smoke')",
        "stream 's': kind must be one of text|binary|telemetry|packet|driver|token (got 'smoke')",
    ),
    Case(
        "m5.stream.element_bits_zero",
        "stream",
        {"kind": "binary", "encoding": "le", "element_bits": 0, "max_window": 64},
        "R21",
        "stream 's': element_bits must be positive (got 0)",
        "stream 's': element_bits must be positive (got 0)",
    ),
    Case(
        "m5.stream.max_window_zero",
        "stream",
        {"kind": "binary", "encoding": "le", "element_bits": 8, "max_window": 0},
        "R21",
        "stream 's': max_window must be positive (got 0)",
        "stream 's': max_window must be positive (got 0)",
    ),
    Case(
        "m5.fsm.legal",
        "fsm",
        _fsm([("s0", False, False), ("s1", True, False)], [("s0", "s1", "claim")]),
    ),
    Case(
        "m5.fsm.state_accepting_and_error",
        "fsm",
        _fsm([("s0", True, True)], []),
        "R21",
        "fsm state 's0': a state is accepting or an error, not both",
        "fsm state 's0': a state is accepting or an error, not both",
    ),
    Case(
        "m5.fsm.transition_to_unknown_state",
        "fsm",
        _fsm([("s0", False, False)], [("s0", "gone", "x")]),
        "R21",
        "fsm transition 's0' -> 'gone' names unknown state 'gone'",
        "fsm transition 't0': @gone does not resolve to a bcir.fsm.state",
    ),
    Case(
        "m5.fsm.duplicate_transition",
        "fsm",
        _fsm([("s0", False, False)], [("s0", "s0", "x"), ("s0", "s0", "x")]),
        "R21",
        "fsm transition on 'x' from state 's0' is declared twice",
        "fsm machine 'fsm': transition on 'x' from @s0 is declared twice",
    ),
    Case(
        "m5.fsm.unknown_start",
        "fsm",
        _fsm([("s0", False, False)], [], start="nowhere"),
        "R21",
        "unknown start state 'nowhere'",
        "fsm machine 'fsm': start_state @nowhere does not name a bcir.fsm.state of this machine",
    ),
    Case(
        "m5.grammar.legal",
        "grammar",
        {
            "start_symbol": "claim",
            "tokens": [["IDENT", "[A-Za-z_][A-Za-z0-9_]*"], ["INT", "[0-9]+"]],
        },
    ),
    Case(
        "m5.grammar.empty_start",
        "grammar",
        {"start_symbol": "", "tokens": [["IDENT", "[A-Za-z_]+"]]},
        "R21",
        "grammar: start_symbol must be a non-empty string (got '')",
        "grammar 'g': start_symbol must be non-empty",
    ),
    Case(
        "m5.token.empty_pattern",
        "grammar",
        {"start_symbol": "claim", "tokens": [["T", ""]]},
        "R21",
        "token rule 'T': pattern must be a non-empty string",
        "token rule 'T': pattern must be non-empty",
    ),
    Case(
        "m5.token.pattern_does_not_compile",
        "grammar",
        {"start_symbol": "claim", "tokens": [["T", "["]]},
        "R21",
        "token rule 'T': pattern does not compile",
        rails=(ORACLE,),
        note="the oracle lexes with Python's re; the dialect carries the pattern as text",
    ),
    Case(
        "m5.grammar.duplicate_token",
        "grammar",
        {"start_symbol": "claim", "tokens": [["T", "a"], ["T", "b"]]},
        "R21",
        "grammar 'g': duplicate token rule names",
        "grammar 'g': duplicate token rule 'T'",
    ),
    # ---- the convolution wire domain (conv.check_conv; the gem.conv op verifier) ---------------
    Case("conv.legal", "conv", _conv(2, 5, 5, 3, 3, 3)),
    Case(
        "conv.output_overflow",
        "conv",
        _conv(1, 1 << 31, 1 << 31, 4, 1, 1),
        "R22",
        "conv: output element count exceeds signed 64-bit range",
        "conv: output element count exceeds signed 64-bit range",
        note="the one-tile conv that wrapped its count to 0 in the GEM lowerer (row 13)",
    ),
    Case(
        "conv.work_overflow",
        "conv",
        _conv(1 << 14, 1 << 20, 1 << 20, 1 << 10, 1, 1),
        "R22",
        "conv: im2col work M*N*K exceeds signed 64-bit range",
        "conv: im2col work M*N*K exceeds signed 64-bit range",
    ),
    Case(
        "conv.stride_zero",
        "conv",
        _conv(2, 5, 5, 3, 3, 3, stride=0),
        "R22",
        "conv: stride must be >= 1; got 0",
        "conv: stride must be >= 1 (got 0)",
    ),
    # ---- MAP/ROP: the derived domain contract (the parsed module, verified on both rails) -----
    Case("map.hbm_only", "map", {"source": _MAP_HBM}),
    Case("map.mmio_write_barriered", "map", {"source": _MAP_MMIO_W}),
    Case(
        "map.mmio_write_default_hazard",
        "map",
        {"source": _MAP_MMIO_W_UNIQUE},
        "R3",
        "claim 1000: MMIO write to RID 5 requires an atomic/barriered hazard contract",
        "R3: claim c1000 MMIO write to @r5 requires an atomic/barriered hazard",
        note="both rails also report R5 (a volatile access needs an ordered hazard)",
        also_laws=("R5",),
        mlir_also=("R5: claim c1000 volatile access requires an atomic/barriered hazard",),
    ),
    Case("rop.hbm_only", "rop", {"source": _ROP_HBM}),
    Case("rop.mmio_read_barriered", "rop", {"source": _ROP_MMIO_R}),
    Case(
        "rop.mmio_read_default_hazard",
        "rop",
        {"source": _ROP_MMIO_R_UNIQUE},
        "R5",
        "claim 1000: volatile access requires an atomic/barriered hazard contract",
        "R5: claim c1000 volatile access requires an atomic/barriered hazard",
    ),
)

# --- the oracle runner ---------------------------------------------------------------------------


def build_module(spec: dict) -> Module:
    m = Module(name=spec.get("name", "m"))
    for r in spec["resources"]:
        m.add_resource(
            Resource(
                rid=r["rid"],
                domain=_DOMAINS[r["domain"]],
                shape=tuple(r["shape"]),
                align=r["align"],
                elem_bytes=r["elem_bytes"],
                access=r["access"],
            )
        )
    claims_by_phase: dict[int, list[Claim]] = {}
    for c in spec["claims"]:
        claims_by_phase.setdefault(c["phase"], []).append(
            Claim(
                id=c["id"],
                opcode=Opcode[c["opcode"]],
                lane=_LANES[c["lane"]],
                stride_class=_STRIDES[c["stride_class"]],
                count=c["count"],
                stride_k=c["stride_k"],
                rd=tuple(c["rd"]),
                wr=tuple(c["wr"]),
                hazard=c["hazard"],
                domain=_DOMAINS[c["domain"]],
                verify=c["verify"],
                bounds=c["bounds"],
                op=c["op"],
                offset=c["offset"],
                volatile=c["volatile"],
            )
        )
    for i, p in enumerate(spec["phases"]):
        # A duplicate phase id is a corpus case: the FIRST phase with the id carries the claims.
        claims = claims_by_phase.get(p["id"], []) if p["sym"] == f"p{p['id']}" else []
        m.add_phase(Phase(phase_id=p["id"], deps=tuple(p["deps"]), claims=claims))
    return m


def spec_from_module(m: Module) -> dict:
    """The rail-neutral spec of a Python module (so a frontend-parsed module renders on the law rail)."""
    resources = [
        _res(
            r.rid,
            shape=r.shape,
            domain=_DOMAIN_SPELL[r.domain],
            align=r.align,
            elem_bytes=r.elem_bytes,
            access=r.access,
        )
        for r in sorted(m.resources.values(), key=lambda r: r.rid)
    ]
    phases = [_phase(p.phase_id, deps=p.deps) for p in m.phases]
    claims = []
    for p in m.phases:
        for c in p.claims:
            claims.append(
                _claim(
                    c.id,
                    p.phase_id,
                    rd=c.rd,
                    wr=c.wr,
                    count=c.count,
                    op=c.op or "vector.add",
                    opcode=c.opcode.name,
                    lane=_LANE_SPELL[c.lane],
                    stride_class=_STRIDE_SPELL[c.stride_class],
                    stride_k=c.stride_k,
                    offset=c.offset,
                    domain=_DOMAIN_SPELL[c.domain],
                    hazard=c.hazard,
                    verify=c.verify,
                    bounds=c.bounds,
                    volatile=c.volatile,
                )
            )
    return _module(resources, phases, claims, name=m.name)


def _oracle_diags(diags) -> Verdict:
    return Verdict(
        ORACLE,
        bool(diags),
        tuple(f"{d.law}: {d.message}" for d in diags),
        tuple(sorted({d.law for d in diags})),
    )


def _refusal(exc: Exception, law: str) -> Verdict:
    return Verdict(ORACLE, True, (f"{type(exc).__name__}: {exc}",), (law,))


def run_oracle(case: Case) -> Verdict:
    """The Python rail's verdict on a case (a constructor refusal is a verdict under the case's law)."""
    from . import verify, verify_address_width, verify_provenance

    kind, spec = case.kind, case.spec
    try:
        if kind == "module":
            return _oracle_diags(verify(build_module(spec)))
        if kind in ("map", "rop"):
            from ..frontends import parse_map, parse_rop_program

            parsed = (
                parse_map(spec["source"]) if kind == "map" else parse_rop_program(spec["source"])
            )
            return _oracle_diags(verify(parsed))
        if kind == "target":
            from ..kbcir.cost import TargetProfile

            TargetProfile(
                cacheline=spec["cacheline"],
                lane_widths=tuple(spec["lane_widths"]),
                elem_bytes=spec["elem_bytes"],
            )
            return Verdict(ORACLE, False)
        if kind == "mmio":
            what = "object address" if spec["op"] in ("rmw", "cas") else "device-register address"
            if spec["triple"] is None:
                return Verdict(ORACLE, False)
            return _oracle_diags(verify_address_width(spec["triple"], spec["addr_bits"], what))
        if kind == "manifest":
            from ..kbcir.provenance import ProvenanceManifest

            doc = {
                "digest": spec["digest"],
                "score": spec["score"],
                "widths": [],
                "artifacts": spec["artifacts"],
                **{k: spec["components"].get(k, 0) for k in _COMPONENTS},
            }
            ProvenanceManifest.from_json(json.dumps(doc))
            return Verdict(ORACLE, False)
        if kind == "portfolio":
            from ..kbcir.portfolio import PolicyPortfolio, PortfolioEntry, ReplayCertificate
            from ..kbcir.weights import PERF, Policy

            portfolio = PolicyPortfolio()
            portfolio.entries = {}
            for slot, gen, certified in spec["entries"]:
                portfolio.entries[slot] = PortfolioEntry(
                    policy=Policy(name=slot, base=PERF.base), gen=gen, certified=certified
                )
            certs = [ReplayCertificate(c, i, e, r) for c, i, e, r in spec["certs"]]
            return _oracle_diags(verify_provenance(portfolio, certificates=certs))
        if kind == "calibration":
            from ..kbcir.cost import TargetProfile
            from ..kbcir.microbench import CalibratedProfile

            table = CalibratedProfile(
                name="cpu",
                cal_gen=spec["cal_gen"],
                samples=5,
                provenance="host",
                strided_q8=spec["strided_q8"],
                random_q8=spec["random_q8"],
            )
            h = TargetProfile(
                name="cpu",
                cal_gen=spec["cal_gen"],
                gather_penalty=spec["gather_penalty"],
                base_overhead=spec["base_overhead"],
                mem_unit=1,
            )
            return _oracle_diags(verify_provenance(None, h=h, table=table))
        if kind == "binary":
            from ..etl.binary import BinaryField, BinaryFormat, BinaryRecord

            fields = tuple(BinaryField(n, o, w, kind=k) for n, o, w, k in spec["fields"])
            record = BinaryRecord("r", fields, size_bits=spec["size_bits"])
            BinaryFormat(
                "f",
                endianness=spec["endianness"],
                alignment_bits=spec["alignment_bits"],
                records=(record,),
            )
            return Verdict(ORACLE, False)
        if kind == "stream":
            from ..etl.events import EventStream

            EventStream(
                "s",
                kind=spec["kind"],
                encoding=spec["encoding"],
                element_bits=spec["element_bits"],
                max_window=spec["max_window"],
            )
            return Verdict(ORACLE, False)
        if kind == "fsm":
            from ..etl.fsm import State, Transducer, Transition

            Transducer(
                [State(n, accepting=a, error=e) for n, a, e in spec["states"]],
                [Transition(s, d, on=o) for s, d, o in spec["transitions"]],
                start=spec["start"],
            )
            return Verdict(ORACLE, False)
        if kind == "grammar":
            from ..etl.parse import Grammar, TokenRule

            Grammar(
                "g", "ebnf", spec["start_symbol"], tuple(TokenRule(n, p) for n, p in spec["tokens"])
            )
            return Verdict(ORACLE, False)
        if kind == "conv":
            from ..kbcir.conv import ConvSpec, check_conv

            s = ConvSpec(**spec)
            if s.stride < 1:
                errs = check_conv(
                    s,
                    (s.in_c, s.in_h, s.in_w),
                    s.dtype,
                    (s.out_c, s.in_c, s.kh, s.kw),
                    (s.out_c, 0, 0),
                    s.dtype,
                )
            else:
                errs = check_conv(
                    s,
                    (s.in_c, s.in_h, s.in_w),
                    s.dtype,
                    (s.out_c, s.in_c, s.kh, s.kw),
                    (s.out_c, s.out_h, s.out_w),
                    s.dtype,
                )
            return Verdict(ORACLE, bool(errs), tuple(errs), (case.law,) if errs else ())
    except ValueError as exc:
        return _refusal(exc, case.law)
    except Exception as exc:  # noqa: BLE001 -- a frontend's own error type is still a verdict
        if type(exc).__name__ in ("MapError", "ParseError"):
            return _refusal(exc, case.law)
        raise
    raise ValueError(f"unknown corpus kind {kind!r}")


# --- the law-rail renderer ------------------------------------------------------------------------


def _claim_line(c: dict) -> str:
    refs = lambda rids: "[" + ", ".join(f"@r{r}" for r in rids) + "]"  # noqa: E731
    vol = ", is_volatile = true" if c["volatile"] else ""
    return (
        f"  bcir.claim @c{c['id']} attributes {{ claim_id = {c['id']} : i32, phase = @{c['phase_sym']}, "
        f'op = "{c["op"]}", reads = {refs(c["rd"])}, writes = {refs(c["wr"])}, count = {c["count"]} : i64, '
        f"lane = #bcir.lane<{c['lane']}>, stride_class = #bcir.stride_class<{c['stride_class']}>, "
        f"stride_k = {c['stride_k']} : i32, offset = {c['offset']} : i64, domain = #bcir.domain<{c['domain']}>, "
        f"hazard = #bcir.hazard<{c['hazard']}>, verify = #bcir.verify<{c['verify']}>, "
        f"bounds = #bcir.bounds<{c['bounds']}>{vol} }} "
        f"{{ %i = bcir.index_range 0 to {max(c['count'], 0)} step 1 }}"
    )


def _render_module(spec: dict) -> list[str]:
    L = [f"bcir.module @{spec.get('name', 'm')} {{"]
    if spec["resources"]:
        L.append("  bcir.registry @RES {")
        for r in spec["resources"]:
            access = ", access = #bcir.access<ham>" if r["access"] == "ham" else ""
            L.append(
                ("    " + _ERR if r["err"] else "    ")
                + f"bcir.resource @r{r['rid']} {{ rid = {r['rid']} : i32, domain_kind = #bcir.domain<{r['domain']}>, "
                f"shape = {_i64_array(r['shape'])}, layout = #bcir.layout<soa>, align = {r['align']} : i32{access} }}"
            )
        L.append("  }")
    for p in spec["phases"]:
        deps = ", ".join(f"@p{d}" for d in p["deps"])
        L.append(
            ("  " + _ERR if p["err"] else "  ")
            + f"bcir.phase @{p['sym']} {{ id = {p['id']} : i32, deps = [{deps}] }}"
        )
    for c in spec["claims"]:
        line = _claim_line(c)
        L.append(("  " + _ERR + line.lstrip()) if c["err"] else line)
    L.append("}")
    return L


def _i64_array(values) -> str:
    """A DenseI64ArrayAttr: `array<i64: 1, 2>`, and `array<i64>` when empty (the parser's spelling)."""
    values = list(values)
    return "array<i64: " + ", ".join(str(v) for v in values) + ">" if values else "array<i64>"


def _capability(triple: str, cacheline=64, lane_widths=(1, 8, 16), elem_bytes=4, extra="") -> str:
    return (
        f'  bcir.target.capability @cpu {{ triple = "{triple}", isa_features = [], '
        f"lane_widths = {_i64_array(lane_widths)}, cacheline = {cacheline} : i32, elem_bytes = {elem_bytes} : i32{extra} }}"
    )


def render(case: Case, *, expectations: bool) -> str:
    """The case as one BCIR-MLIR module; with `expectations`, the `expected-error` marker precedes
    the offending line (the -verify-diagnostics projection); without, the sentinel is dropped."""
    kind, spec = case.kind, case.spec
    if kind == "module":
        L = _render_module(spec)
    elif kind in ("map", "rop"):
        from ..frontends import parse_map, parse_rop_program

        parsed = parse_map(spec["source"]) if kind == "map" else parse_rop_program(spec["source"])
        mspec = spec_from_module(parsed)
        if case.law:
            for c in mspec["claims"]:
                c["err"] = True  # the frontends' cases attach their diagnostic to the (one) claim
        L = _render_module(mspec)
    elif kind == "target":
        L = [
            "bcir.module @m {",
            (_ERR if case.law else "")
            + _capability(
                "x86_64-avx512", spec["cacheline"], spec["lane_widths"], spec["elem_bytes"]
            ).lstrip(),
            "}",
        ]
        if case.law:
            L[1] = "  " + L[1]
    elif kind == "mmio":
        bits = spec["addr_bits"]
        L = ["bcir.module @m {"]
        if spec["triple"] is not None:
            L.append(_capability(spec["triple"]))
        err = "  " + _ERR if case.law else "  "
        if spec["op"] == "load":
            L += [
                f"  func.func @f(%addr: i{bits}) -> i32 {{",
                f"  {err}%v = bcir.volatile_load %addr : i{bits} -> i32",
                "    return %v : i32",
                "  }",
            ]
        elif spec["op"] == "store":
            L += [
                f"  func.func @f(%v: i32, %addr: i{bits}) {{",
                f"  {err}bcir.volatile_store %v, %addr : i32, i{bits}",
                "    return",
                "  }",
            ]
        elif spec["op"] == "rmw":
            L += [
                f"  func.func @f(%v: i32, %addr: i{bits}) -> i32 {{",
                f'  {err}%o = bcir.atomic_rmw "add" %v, %addr : i32, i{bits} -> i32',
                "    return %o : i32",
                "  }",
            ]
        else:
            L += [
                f"  func.func @f(%e: i32, %d: i32, %addr: i{bits}) -> i32 {{",
                f"  {err}%o = bcir.atomic_cas %e, %d, %addr : i32, i32, i{bits} -> i32",
                "    return %o : i32",
                "  }",
            ]
        L.append("}")
    elif kind == "manifest":
        names = spec["names"] if spec["names"] is not None else [a[0] for a in spec["artifacts"]]
        gens = spec["gens"] if spec["gens"] is not None else [a[1] for a in spec["artifacts"]]
        arts = ""
        if names or gens:
            arts = (
                ", artifact_names = ["
                + ", ".join(f'"{n}"' for n in names)
                + "], artifact_gens = array<i64: "
                + ", ".join(str(g) for g in gens)
                + ">"
            )
        L = ["bcir.module @m {"]
        if spec["with_claims"]:
            L += [
                "  bcir.registry @RES {",
                "    bcir.resource @r1 { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }",
                "  }",
                "  bcir.phase @p0 { id = 0 : i32, deps = [] }",
                _claim_line(_claim()),
            ]
        if spec["with_theta"]:
            L.append("  bcir.kbcir.theta @theta { thermal = 0 : i32 }")
        comps = "".join(f", {k} = {v} : i64" for k, v in spec["components"].items())
        n_art = len(spec["artifacts"]) if spec["names"] is None else len(names)
        L.append(
            ("  " + _ERR if case.law else "  ")
            + f"bcir.kbcir.provenance_manifest @man {{ digest = {spec['digest']} : i64, score = {spec['score']} : i64, "
            f"n_artifacts = {n_art} : i64, reproduced = true{comps}{arts} }}"
        )
        L.append("}")
    elif kind == "portfolio":
        L = ["bcir.module @m {"]
        w = ", ".join(str(x) for x in _LATENCY_WEIGHTS)
        for slot, _gen, _cert in spec["entries"]:
            L.append(
                f"  bcir.kbcir.policy @{slot} {{ mode = #bcir.policy_mode<latency>, weights = array<i64: {w}> }}"
            )
        for i, (cand, inc, ep, reg) in enumerate(spec["certs"]):
            L.append(
                f"  bcir.kbcir.replay_certificate @rc{i} {{ candidate = @{cand}, incumbent = @{inc}, episodes = {ep} : i64, regressions = {reg} : i64, admitted = {'true' if ep >= 1 and reg == 0 else 'false'} }}"
            )
        pols = ", ".join(f"@{s}" for s, _, _ in spec["entries"])
        gens = ", ".join(str(g) for _, g, _ in spec["entries"])
        cert = ", ".join("1" if c else "0" for _, _, c in spec["entries"])
        L.append(
            ("  " + _ERR if case.law else "  ")
            + f"bcir.kbcir.portfolio @pf {{ policies = [{pols}], gens = array<i64: {gens}>, certified = array<i64: {cert}> }}"
        )
        L.append("}")
    elif kind == "calibration":
        extra = f", gather_penalty = {spec['gather_penalty']} : i32, base_overhead = {spec['base_overhead']} : i32, mem_unit = 1 : i32, cal_gen = {spec['cal_gen']} : i64"
        cap = _capability("x86_64-avx512", extra=extra)
        L = [
            "bcir.module @m {",
            ("  " + _ERR + cap.lstrip()) if case.law else cap,
            f'  bcir.kbcir.calibration @cal {{ target = @cpu, cal_gen = {spec["cal_gen"]} : i64, samples = 5 : i64, provenance = "host", stream_q8 = 256 : i64, strided_q8 = {spec["strided_q8"]} : i64, random_q8 = {spec["random_q8"]} : i64, compute_q8 = 256 : i64 }}',
            "}",
        ]
    elif kind == "binary":
        fields = spec["fields"]
        bad_field = next(
            (f for f in fields if f[1] < 0 or f[2] < 1 or f[3] not in ("u", "s", "f", "bytes")),
            None,
        )
        fmt_bad = (
            spec["endianness"] not in ("little", "big", "le", "be") or spec["alignment_bits"] < 1
        )
        L = ["bcir.module @m {"]
        L.append(
            ("  " + _ERR if (case.law and fmt_bad) else "  ")
            + f'bcir.binary.format @f attributes {{ endianness = "{spec["endianness"]}", alignment_bits = {spec["alignment_bits"]} : i64 }} {{'
        )
        for n, o, w, k in fields:
            L.append(
                (
                    "    " + _ERR
                    if (case.law and bad_field is not None and [n, o, w, k] == bad_field)
                    else "    "
                )
                + f'bcir.binary.field @{n} {{ name = "{n}", offset_bits = {o} : i64, width_bits = {w} : i64, kind = "{k}", semantic = "" }}'
            )
        refs = (
            spec["record_fields"] if spec["record_fields"] is not None else [f[0] for f in fields]
        )
        rec_bad = case.law and bad_field is None and not fmt_bad
        L.append(
            ("    " + _ERR if rec_bad else "    ")
            + f'bcir.binary.record @r {{ name = "r", fields = [{", ".join("@" + r for r in refs)}], size_bits = {spec["size_bits"]} : i64 }}'
        )
        L += ["  }", "}"]
    elif kind == "stream":
        L = [
            "bcir.module @m {",
            ("  " + _ERR if case.law else "  ")
            + f'bcir.event.stream @s {{ kind = "{spec["kind"]}", encoding = "{spec["encoding"]}", element_bits = {spec["element_bits"]} : i64, max_window = {spec["max_window"]} : i64 }}',
            "}",
        ]
    elif kind == "fsm":
        states, transitions = spec["states"], spec["transitions"]
        bad_state = next((s for s in states if s[1] and s[2]), None)
        names = {s[0] for s in states}
        bad_tx = next((t for t in transitions if t[0] not in names or t[1] not in names), None)
        seen: set = set()
        dup_tx = None
        for t in transitions:
            if (t[0], t[2]) in seen:
                dup_tx = t
            seen.add((t[0], t[2]))
        machine_bad = (
            case.law
            and bad_state is None
            and bad_tx is None
            and (spec["start"] not in names or dup_tx is not None)
        )
        L = [
            "bcir.module @m {",
            '  bcir.event.stream @src { kind = "token", encoding = "utf8", element_bits = 8 : i64, max_window = 64 : i64 }',
        ]
        L.append(
            ("  " + _ERR if machine_bad else "  ")
            + f'bcir.fsm.machine @fsm attributes {{ kind = "transducer", input_stream = @src, start_state = @{spec["start"]} }} {{'
        )
        for n, a, e in states:
            L.append(
                (
                    "    " + _ERR
                    if (case.law and bad_state is not None and [n, a, e] == bad_state)
                    else "    "
                )
                + f"bcir.fsm.state @{n} {{ accepting = {'true' if a else 'false'}, error = {'true' if e else 'false'} }}"
            )
        for i, (s, d, o) in enumerate(transitions):
            L.append(
                (
                    "    " + _ERR
                    if (case.law and bad_tx is not None and [s, d, o] == bad_tx)
                    else "    "
                )
                + f'bcir.fsm.transition @t{i} {{ from = @{s}, to = @{d}, on = "{o}", guard = "", action = "" }}'
            )
        L += ["  }", "}"]
    elif kind == "grammar":
        tokens = spec["tokens"]
        bad_tok = next((t for t in tokens if not t[1]), None)
        names = [t[0] for t in tokens]
        grammar_bad = case.law and bad_tok is None
        L = [
            "bcir.module @m {",
            ("  " + _ERR if grammar_bad else "  ")
            + f'bcir.parse.grammar @g attributes {{ syntax = "ebnf", start_symbol = "{spec["start_symbol"]}" }} {{',
        ]
        for n, p in tokens:
            pat = p.replace("\\", "\\\\").replace('"', '\\"')
            L.append(
                (
                    "    " + _ERR
                    if (case.law and bad_tok is not None and [n, p] == bad_tok)
                    else "    "
                )
                + f'bcir.parse.token @{n} {{ pattern = "{pat}", skip = false, precedence = 0 : i32 }}'
            )
        L += ["  }", "}"]
    elif kind == "conv":
        from ..kbcir.conv import ConvSpec

        s = ConvSpec(**spec)
        if s.stride >= 1:
            oh, ow = s.out_h, s.out_w
            m, n, k = s.gemm_dims
        else:  # the derived dims are meaningless; the op verifier refuses the stride first
            oh = ow = 1
            m, n, k = 1, s.out_c, s.in_c * s.kh * s.kw
        L = [
            "bcir.module @m {",
            ("  " + _ERR if case.law else "  ")
            + f"bcir.gem.conv @cv {{ in_c = {s.in_c} : i64, in_h = {s.in_h} : i64, in_w = {s.in_w} : i64, out_c = {s.out_c} : i64, "
            f'kh = {s.kh} : i64, kw = {s.kw} : i64, stride = {s.stride} : i64, pad = {s.pad} : i64, dtype = "{s.dtype}", '
            f"out_h = {oh} : i64, out_w = {ow} : i64, gemm_m = {m} : i64, gemm_n = {n} : i64, gemm_k = {k} : i64, "
            f'strategy = "direct", tile_m = {m} : i64, tile_n = {n} : i64, tile_k = {k} : i64, loop_order = "ijk", '
            f"compute_cost = 0 : i64, mem_cost = 0 : i64, bottleneck = 0 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }}",
            "}",
        ]
    else:
        raise ValueError(f"unknown corpus kind {kind!r}")

    out = []
    for line in L:
        if _ERR in line:
            indent = line[: line.index(_ERR)]
            body = line.replace(_ERR, "")
            if expectations:
                expected = case.mlir_expected
                assert expected, f"{case.name}: an offending line needs an expected diagnostic"
                for i, msg in enumerate(expected):
                    out.append(f"{indent}// expected-error @+{len(expected) - i} {{{{{msg}}}}}")
            out.append(body)
        else:
            out.append(line)
    return "\n".join(out) + "\n"


# --- the law-rail runner ------------------------------------------------------------------------


def find_bcir_opt() -> str | None:
    env = os.environ.get("BCIR_OPT")
    if env and os.path.exists(env):
        return env
    root = _ROOT / "build" / "mlir-build"
    if root.is_dir():
        for dirpath, _dirs, files in os.walk(root):
            if "bcir-opt" in files:
                return os.path.join(dirpath, "bcir-opt")
    return None


_LAW_RE = re.compile(r"error: (R\d+):")


def run_mlir(case: Case, bcir_opt: str) -> Verdict:
    """Drive `bcir-opt -bcir-verify` over the rendered case; every `error:` line is a message."""
    proc = subprocess.run(
        [bcir_opt, "-bcir-verify"],
        input=render(case, expectations=False),
        capture_output=True,
        text=True,
    )
    errors = tuple(
        line.split("error: ", 1)[1] for line in proc.stderr.splitlines() if "error: " in line
    )
    laws = tuple(
        sorted({m.group(1) for line in errors for m in [_LAW_RE.search("error: " + line)] if m})
    )
    return Verdict(MLIR, proc.returncode != 0 or bool(errors), errors, laws)


# --- the comparison ------------------------------------------------------------------------------


def findings(case: Case, verdicts: dict[str, Verdict]) -> list[Finding]:
    """Every way a rail's verdict can disagree with the corpus, as findings (never a pass)."""
    out: list[Finding] = []
    for rail in case.rails:
        v = verdicts.get(rail)
        if v is None:
            out.append(Finding(case.name, rail, "no-verdict", "the rail produced no verdict"))
            continue
        expected_all = (case.oracle,) if rail == ORACLE else case.mlir_expected
        expected = expected_all[0] if expected_all else ""
        if case.legal:
            if v.refused:
                out.append(Finding(case.name, rail, "refused-legal", "; ".join(v.messages)[:400]))
            continue
        if not v.refused:
            out.append(
                Finding(case.name, rail, "admitted-illegal", f"expected {case.law}: {expected!r}")
            )
            continue
        for want in expected_all:
            if want and not any(want in m for m in v.messages):
                out.append(
                    Finding(
                        case.name, rail, "message", f"expected {want!r}, got {list(v.messages)[:3]}"
                    )
                )
        if rail == MLIR and len(v.messages) > len(expected_all):
            out.append(
                Finding(
                    case.name,
                    rail,
                    "message",
                    f"{len(v.messages)} diagnostics for {len(expected_all)} expected: {list(v.messages)[:4]}",
                )
            )
        allowed = {case.law, *case.also_laws}
        named = {law for law in v.laws if law}
        if named and not named <= allowed:
            out.append(
                Finding(
                    case.name,
                    rail,
                    "law",
                    f"reported {sorted(named)}, corpus allows {sorted(allowed)}",
                )
            )
    return out


def run(bcir_opt: str | None, cases=CASES) -> tuple[list[Finding], dict]:
    """Both runners over the corpus. Returns (findings, counts)."""
    found: list[Finding] = []
    counts = {"cases": 0, "oracle": 0, "mlir": 0}
    for case in cases:
        counts["cases"] += 1
        verdicts: dict[str, Verdict] = {}
        rails = list(case.rails)
        if ORACLE in rails:
            verdicts[ORACLE] = run_oracle(case)
            counts["oracle"] += 1
        if MLIR in rails:
            if bcir_opt:
                verdicts[MLIR] = run_mlir(case, bcir_opt)
                counts["mlir"] += 1
            else:
                rails.remove(MLIR)  # no toolchain: the law rail is not judged, never faked
        found += findings(replace(case, rails=tuple(rails)), verdicts)
    return found, counts


# --- the fixture projection -----------------------------------------------------------------------

_HEADER = """// RUN: bcir-opt -bcir-verify -verify-diagnostics -split-input-file %s
//
// GENERATED by `python -m bcir.verify.structural_corpus --emit` -- do not edit by hand.
// The shared structural-law corpus (S0-6): every law-rail case of bcir/verify/structural_corpus.py,
// rendered with its expected diagnostic. The oracle runner over the SAME cases is
// bcir/tests/test_structural_corpus.py (the quick tier), which also refuses drift between this
// file and the corpus (`--check`). Widths, alignments, shapes and strides; phase identity; the
// isolated-domain rule; the target descriptor; the address width under a declared target; the
// R13 manifest, portfolio and calibration records; the M5 descriptors; the convolution wire
// domain; the MAP/ROP derived domain contract -- one corpus, two runners, every mismatch a finding.
"""


def emit_fixture(cases=CASES) -> str:
    parts = [_HEADER]
    for case in cases:
        if MLIR not in case.rails:
            continue
        head = f"// {case.name} [{case.law or 'legal'}]" + (f" -- {case.note}" if case.note else "")
        parts.append(head + "\n" + render(case, expectations=True))
    return "\n// -----\n\n".join(parts)


def check_fixture(path: Path = _ROOT / FIXTURE) -> list[str]:
    """[] when the committed fixture is the emitted corpus; else the reasons."""
    if not path.exists():
        return [f"{path} does not exist (run --emit)"]
    if path.read_text(encoding="utf-8") != emit_fixture():
        return [f"{path} drifted from bcir/verify/structural_corpus.py (run --emit)"]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--emit", action="store_true", help="write the law-rail fixture")
    parser.add_argument("--check", action="store_true", help="refuse fixture drift")
    parser.add_argument("--run", action="store_true", help="run both rails and report findings")
    parser.add_argument(
        "--bcir-opt", default=None, help="the bcir-opt binary (default: found under build/)"
    )
    args = parser.parse_args(argv)
    rc = 0
    if args.emit:
        (_ROOT / FIXTURE).write_text(emit_fixture(), encoding="utf-8")
        print(
            f"structural_corpus: wrote {FIXTURE} ({sum(1 for c in CASES if MLIR in c.rails)} law-rail cases)"
        )
    if args.check:
        reasons = check_fixture()
        print("structural_corpus: fixture " + ("current" if not reasons else "; ".join(reasons)))
        rc |= 1 if reasons else 0
    if args.run:
        bo = args.bcir_opt or find_bcir_opt()
        found, counts = run(bo)
        print(
            f"structural_corpus: {counts['cases']} cases, oracle {counts['oracle']}, mlir {counts['mlir']}"
            + ("" if bo else " (no bcir-opt: the law rail was not judged)")
            + f", findings {len(found)}"
        )
        for f in found:
            print(f"  {f.rail:<7} {f.kind:<17} {f.case:<44} {f.detail}")
        rc |= 1 if found else 0
    if not (args.emit or args.check or args.run):
        parser.print_help()
        return 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
