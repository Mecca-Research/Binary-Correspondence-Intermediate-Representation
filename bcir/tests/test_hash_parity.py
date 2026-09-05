"""S0-1 / staged plan S0-D: the two cross-rail content hashes cover every plan-affecting input,
and the law rail recomputes them from the IR -- the differential regression.

Row 4 of the 2026-07/08 assessment: provenance equality did not imply identical plans.
`hash_target` omitted the memory hierarchy (scaling the DRAM tier's factors by 32 moved
`vector_add(4096)` from 31,232 to 983,552 under one hash) and `hash_module` sorted claims by
id (declaring `a, b` as `b, a` moved a chain from 9,216 to 10,496 under one hash). Both are
CROSS-RAIL hashes: `-bcir-verify` recomputes them field for field for R13's manifest check,
so widening them meant dialect attributes for the tiers (`target.capability`
`mem_tier_names` / `mem_tier_values`), the matching C++ walks, the emitter writing every
hashed field, and the pinned constants -- on both rails in one commit. This module is the
regression that keeps the two rails agreeing: the oracle half always runs; the law-rail half
drives `bcir-opt` over emitted modules with their real manifests and self-skips without the
toolchain.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import replace

from bcir.examples import vector_add
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import MemoryHierarchy, MemTier, Theta, Tier
from bcir.kbcir.provenance import build_manifest, hash_module, hash_target
from bcir.kbcir.weights import PERF
from bcir.lower.mlir import to_mlir
from bcir.model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass

_AVX = TARGETS["x86_avx512"]
_COOL = Theta.cool()
_TIER_ATTRS = re.compile(r", mem_tier_names = \[[^\]]*\], mem_tier_values = array<i64: [^>]*>")


def _chain(order: str) -> Module:
    """Two dependent claims declared in `order` ('ab' or 'ba'); same ids either way."""
    m = Module(name="chain")
    for rid in (1, 2, 3):
        m.add_resource(Resource(rid=rid, shape=(1024,)))
    claims = {
        "a": Claim(
            id=1,
            opcode=Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=1024,
            rd=(1,),
            wr=(2,),
            op="vector.add",
        ),
        "b": Claim(
            id=2,
            opcode=Opcode.MUL,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=1024,
            rd=(2,),
            wr=(3,),
            op="vector.mul",
        ),
    }
    m.add_phase(Phase(phase_id=0, claims=[claims[k] for k in order]))
    return m


def _two_phase_module() -> Module:
    """Two phases, one of them claim-less, ids declared out of dependency order."""
    m = Module(name="two_phase")
    for rid in (1, 2):
        m.add_resource(Resource(rid=rid, shape=(256,)))
    m.add_phase(Phase(phase_id=1, deps=()))
    m.add_phase(
        Phase(
            phase_id=0,
            deps=(1,),
            claims=[
                Claim(
                    id=7,
                    opcode=Opcode.ADD,
                    lane=Lane.U,
                    stride_class=StrideClass.UNIT,
                    count=256,
                    rd=(1,),
                    wr=(2,),
                    op="vector.add",
                )
            ],
        )
    )
    return m


def _scaled_dram(h, factor: int = 32):
    tiers = list(h.mem.tiers)
    dram = next(i for i, t in enumerate(tiers) if t.name == "DRAM")
    tiers[dram] = Tier(
        "DRAM",
        tiers[dram].latency_cyc,
        tiers[dram].bw_factor * factor,
        tiers[dram].lat_factor * factor,
    )
    return replace(h, mem=MemoryHierarchy(tuple(tiers)))


def _own_hierarchy() -> MemoryHierarchy:
    """A hardware channel's own tiers: names the MemTier enum does not know are data too."""
    return MemoryHierarchy((Tier("sram", 2, 8, 8, capacity=1 << 18), Tier("DRAM", 200, 256, 256)))


def _tier_attrs(h) -> str:
    return (
        "mem_tier_names = [" + ", ".join(f'"{n}"' for n in h.mem.tier_names()) + "], "
        "mem_tier_values = array<i64: " + ", ".join(str(x) for x in h.mem.tier_values()) + ">"
    )


# --- the oracle half (always runs) -------------------------------------------------------------


def test_hash_target_separates_two_memory_hierarchies() -> None:
    """The audit's collision: one tier scaled, the score moves thirty-fold, and now the hash
    moves with it. Every other field is identical, so it is the tiers that separate them."""
    scaled = _scaled_dram(_AVX)
    base_score = optimize(vector_add(4096), _AVX, _COOL, PERF).score
    scaled_score = optimize(vector_add(4096), scaled, _COOL, PERF).score
    assert scaled_score > 20 * base_score, (base_score, scaled_score)
    assert hash_target(_AVX) != hash_target(scaled)
    assert hash_target(_AVX) == hash_target(replace(_AVX, mem=MemoryHierarchy.default()))


def test_hash_module_separates_two_declared_claim_orders() -> None:
    """The audit's other collision: the same two claims declared in the other order plan to a
    different score, and now hash differently. Sorted content is identical by construction."""
    ab, ba = _chain("ab"), _chain("ba")
    assert optimize(ab, _AVX, _COOL, PERF).score != optimize(ba, _AVX, _COOL, PERF).score
    assert hash_module(ab) != hash_module(ba)
    assert [c.id for c in sorted(ab.phases[0].claims, key=lambda c: c.id)] == [
        c.id for c in sorted(ba.phases[0].claims, key=lambda c: c.id)
    ]
    # the phase-dependency SET stays a set: spelling deps in another order is not a new graph
    m1, m2 = _two_phase_module(), _two_phase_module()
    m2.phases[1].deps = tuple(reversed(m1.phases[1].deps))
    assert hash_module(m1) == hash_module(m2)


def test_every_shipped_profile_carries_the_default_hierarchy_the_law_rail_pins() -> None:
    """`hashTargetFromIR` falls back to a pinned copy of MemoryHierarchy.default() when a
    capability carries no `mem_tier_names` / `mem_tier_values`; that table is only right if
    every shipped profile really carries the default -- and the emitted arrays are the table,
    name for name and value for value (BCIRVerifyPass.cpp kDefaultMemTier*)."""
    names = MemoryHierarchy.default().tier_names()
    values = MemoryHierarchy.default().tier_values()
    assert names == ("L1", "L2", "L3", "DRAM", "HBM", "CXL", "SSD")
    assert [MemTier[n].value for n in names] == list(range(7))
    assert values == (
        4, 16, 16, 0,
        12, 32, 48, 0,
        40, 96, 96, 0,
        200, 256, 256, 0,
        160, 64, 192, 0,
        350, 384, 512, 0,
        5000, 1024, 4096, 0,
    )  # fmt: skip
    for name, h in TARGETS.items():
        assert (h.mem.tier_names(), h.mem.tier_values()) == (names, values), name
        assert _tier_attrs(h) in to_mlir(vector_add(256), h, _COOL, PERF), name


def test_a_hierarchy_is_data_the_verifier_can_recompute_from() -> None:
    """Construction-time rules, the twins of the tier-array verifier; and a channel's own tier
    (a name the MemTier enum does not know) is data too -- it hashes and emits like the seven."""
    for bad in (
        (Tier("DRAM", 200, 256, 256), Tier("DRAM", 200, 256, 256)),  # declared twice
        (Tier("", 1, 1, 1),),  # an unnamed tier
        (Tier("DRAM", 200, 0, 256),),  # a zero ratio
        (Tier("DRAM", -1, 256, 256),),  # negative latency
    ):
        try:
            MemoryHierarchy(bad)
        except ValueError:
            continue
        raise AssertionError(f"MemoryHierarchy accepted {bad}")
    own = _own_hierarchy()
    assert own.tier_names() == ("sram", "DRAM")
    assert hash_target(replace(_AVX, mem=own)) != hash_target(_AVX)
    assert '"sram"' in to_mlir(vector_add(256), replace(_AVX, mem=own), _COOL, PERF)


# --- the law-rail half (self-skips without bcir-opt) ---------------------------------------------


def _find_bcir_opt():
    env = os.environ.get("BCIR_OPT")
    if env and os.path.exists(env):
        return env
    root = os.path.join(os.path.dirname(__file__), "..", "..", "build", "mlir-build")
    for dirpath, _dirs, files in os.walk(os.path.normpath(root)) if os.path.isdir(root) else []:
        if "bcir-opt" in files:
            return os.path.join(dirpath, "bcir-opt")
    return None


def _module_with_manifest(module: Module, h, theta=_COOL, policy=PERF) -> str:
    """The emitted plan IR plus the real manifest of that plan: every component hash the
    oracle computed, for the law rail to recompute and cross-check."""
    result = optimize(module, h, theta, policy)
    man = build_manifest(module, h, theta, policy)
    text = to_mlir(module, h, theta, policy, result=result)
    manifest = (
        f"  bcir.kbcir.provenance_manifest @man {{ digest = {man.digest} : i64, "
        f"score = {man.score} : i64, n_artifacts = 0 : i64, reproduced = true, "
        f"m_module = {man.m_module} : i64, m_target = {man.m_target} : i64, "
        f"m_theta = {man.m_theta} : i64, m_policy = {man.m_policy} : i64 }}\n"
    )
    assert text.endswith("}\n")
    return text[:-2] + manifest + "}\n"


def _verify(bo: str, text: str) -> list[str]:
    proc = subprocess.run([bo, "-bcir-verify"], input=text, capture_output=True, text=True)
    return [line.split("error: ", 1)[1] for line in proc.stderr.splitlines() if "error: " in line]


def _without_tiers(text: str) -> str:
    out, n = _TIER_ATTRS.subn("", text)
    assert n == 1, "the emitted capability carries the two tier attributes exactly once"
    return out


def test_law_rail_recomputes_both_widened_hashes_from_the_ir() -> None:
    """Every shipped target x four modules: the emitted IR carries its own manifest and
    `-bcir-verify` accepts it -- the law rail's hashTargetFromIR (tiers included) and
    hashModuleFromIR (declared order) reproduce the oracle's m_target and m_module exactly."""
    bo = _find_bcir_opt()
    if not bo:
        return
    for name, h in TARGETS.items():
        for module in (vector_add(1024), _chain("ab"), _chain("ba"), _two_phase_module()):
            errors = _verify(bo, _module_with_manifest(module, h))
            assert errors == [], (name, module.name, errors)


def test_law_rail_refuses_a_manifest_bound_to_the_other_claim_order() -> None:
    """Swap the two claims' textual order in the emitted IR while keeping the manifest of the
    declared order: the recomputed m_module no longer matches (RED on the parent: the sort by
    id made the two orders one hash, so the swap verified clean)."""
    bo = _find_bcir_opt()
    if not bo:
        return
    text = _module_with_manifest(_chain("ab"), _AVX)
    claims = re.findall(r"^  bcir\.claim @c\d+ .*?\n", text, re.M | re.S)
    assert len(claims) == 2, claims
    swapped = (
        text.replace(claims[0], "\x00").replace(claims[1], claims[0]).replace("\x00", claims[1])
    )
    assert swapped != text
    errors = _verify(bo, swapped)
    assert any(e.startswith("R13: manifest m_module") for e in errors), errors
    assert _verify(bo, text) == []


def test_law_rail_refuses_a_manifest_bound_to_the_other_hierarchy() -> None:
    """Scale the DRAM tier in the emitted tier values while keeping the default-hierarchy
    manifest: the recomputed m_target no longer matches (RED on the parent: the attributes
    did not exist and the tiers were not hashed, so any hierarchy verified against any
    manifest). A capability without the attributes hashes as the default hierarchy, the
    manifest of a scaled profile is refused against it, and a channel's own tier names travel
    and hash on the law rail too."""
    bo = _find_bcir_opt()
    if not bo:
        return
    default_text = _module_with_manifest(vector_add(1024), _AVX)
    default_values = ", ".join(str(x) for x in MemoryHierarchy.default().tier_values())
    scaled_values = ", ".join(str(x) for x in _scaled_dram(_AVX).mem.tier_values())
    assert f"mem_tier_values = array<i64: {default_values}>" in default_text
    errors = _verify(bo, default_text.replace(default_values, scaled_values))
    assert any(e.startswith("R13: manifest m_target") for e in errors), errors
    absent = _without_tiers(default_text)
    assert "mem_tier" not in absent
    assert _verify(bo, absent) == []
    scaled_text = _module_with_manifest(vector_add(1024), _scaled_dram(_AVX))
    assert _verify(bo, scaled_text) == []
    errors = _verify(bo, _without_tiers(scaled_text))
    assert any(e.startswith("R13: manifest m_target") for e in errors), errors
    own = replace(_AVX, mem=_own_hierarchy())
    assert _verify(bo, _module_with_manifest(vector_add(1024), own)) == []


def test_law_rail_refuses_a_malformed_hierarchy() -> None:
    bo = _find_bcir_opt()
    if not bo:
        return
    text = _module_with_manifest(vector_add(1024), _AVX)
    values = ", ".join(str(x) for x in MemoryHierarchy.default().tier_values())
    names = ", ".join(f'"{n}"' for n in MemoryHierarchy.default().tier_names())
    assert values in text and names in text
    cases = (
        (text.replace(values, values + ", 3"), "mem_tier_values must hold four values per named"),
        (text.replace("200, 256, 256, 0", "200, 0, 256, 0"), "bw_factor and lat_factor must be"),
        (text.replace(names, names.replace('"SSD"', '"L1"')), "declared twice"),
        (text.replace(names, names.replace('"SSD"', '""')), "non-empty string"),
        (_TIER_ATTRS.sub(', mem_tier_names = ["L1"]', text), "declared together"),
    )
    for bad_text, message in cases:
        errors = _verify(bo, bad_text)
        assert any(message in e for e in errors), (message, errors)
