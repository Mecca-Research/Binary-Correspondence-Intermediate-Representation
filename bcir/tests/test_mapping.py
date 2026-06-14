"""Modular Mapping Functions: objective-support preservation (R12 refinement) and
the commutativity / path-independence law Λ∘Ψ=Φ (R12 parity).

A legal map preserves where the objective matters -- f(Supp(J)) ⊆ Supp(J') -- and
equivalent conversion paths must agree. These pin both laws.
"""

from bcir.examples import vector_add
from bcir.kbcir import TARGETS
from bcir.kbcir.cost import Theta, CostVector
from bcir.kbcir.mapping import (
    CommutingSquare,
    MappingFunction,
    identity_dimmap,
    support,
)
from bcir.kbcir.provenance import ProvenanceManifest, build_manifest
from bcir.verify import verify_commutativity, verify_support_preservation

AVX = TARGETS["x86_avx512"]
COOL = Theta.cool()


def _laws(diags):
    return {d.law for d in diags}


# --- objective support Supp(J) ---------------------------------------------------

def test_support_is_the_set_of_mattering_dimensions():
    J = CostVector.of(compute=10, memory=5, thermal=3, security=2)
    assert support(J) == frozenset({"compute", "memory", "thermal", "security"})
    assert support(CostVector.zero()) == frozenset()


# --- R12 refinement: f(Supp(J)) ⊆ Supp(J') ---------------------------------------

def test_a_lowering_that_preserves_support_is_legal():
    J = CostVector.of(compute=10, memory=5, thermal=3, security=2)
    Jp = CostVector.of(compute=8, memory=4, thermal=1, security=1)   # rescaled, kept
    assert verify_support_preservation(J, Jp) == []


def test_silently_dropping_a_mattering_dimension_is_R12():
    J = CostVector.of(compute=10, memory=5, thermal=3, security=2)
    Jd = CostVector.of(compute=8, memory=4, thermal=1)               # security -> 0
    diags = verify_support_preservation(J, Jd)
    assert "R12" in _laws(diags)
    assert any("security" in d.message for d in diags)


def test_an_explicit_discharge_makes_the_drop_legal():
    J = CostVector.of(compute=10, security=2)
    Jd = CostVector.of(compute=8)                                    # security dropped
    f = MappingFunction(name="discharge_sec", discharged=frozenset({"security"}))
    assert verify_support_preservation(J, Jd, f) == []


def test_growing_support_is_always_legal():
    # A map may add support (model a cost the source ignored); only dropping is illegal.
    J = CostVector.of(compute=10)
    Jp = CostVector.of(compute=8, thermal=2)
    assert verify_support_preservation(J, Jp) == []


def test_a_dimension_remap_carries_support_to_its_image():
    # f sends compute -> power; the law tracks the image, not the name.
    J = CostVector.of(compute=10)
    Jp = CostVector.of(power=8)
    dm = identity_dimmap()
    dm["compute"] = "power"
    f = MappingFunction(name="compute_to_power", dim_map=dm)
    assert verify_support_preservation(J, Jp, f) == []
    # ... but if the image dim is also absent, it is a dropped support.
    assert "R12" in _laws(verify_support_preservation(J, CostVector.of(memory=8), f))


# --- R12 parity: Λ∘Ψ = Φ ---------------------------------------------------------

def test_two_paths_that_agree_commute():
    sq = CommutingSquare("ok", phi=lambda x: x + 3,
                         psi=lambda x: x + 1, lam=lambda y: y + 2)
    assert verify_commutativity(sq, [0, 1, 2, 10]) == []


def test_disagreeing_paths_are_R12():
    sq = CommutingSquare("bad", phi=lambda x: x + 3,
                         psi=lambda x: x + 1, lam=lambda y: y + 5)
    diags = verify_commutativity(sq, [0, 1, 2])
    assert "R12" in _laws(diags) and len(diags) == 3


def test_manifest_json_round_trip_is_a_commuting_square():
    # The parity discipline generalized: serialize-then-parse must equal identity
    # (Λ∘Ψ = Φ with Φ = id). This is the same law as oracle<->MLIR parity.
    m = vector_add(1024)
    man = build_manifest(m, AVX, COOL)
    sq = CommutingSquare("manifest_json", phi=lambda x: x,
                         psi=lambda x: x.to_json(), lam=ProvenanceManifest.from_json)
    assert verify_commutativity(sq, [man]) == []


def test_a_custom_equality_can_be_supplied():
    # Paths that agree only up to an equivalence (here, modulo 10) commute under
    # the supplied eq but not under default ==.
    sq = CommutingSquare("mod10", phi=lambda x: x,
                         psi=lambda x: x + 10, lam=lambda y: y)
    assert verify_commutativity(sq, [3, 7], eq=lambda a, b: a % 10 == b % 10) == []
    assert "R12" in _laws(verify_commutativity(sq, [3, 7]))
