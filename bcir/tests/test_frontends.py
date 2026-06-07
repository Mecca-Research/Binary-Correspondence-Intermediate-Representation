"""CT3: ROP (declarative) + MAP (macro-assembly) front-end tests."""

from bcir.frontends import parse_map, parse_rop_program
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.verify import is_legal

_ROP = """
module vadd {
  resource A { rid 10 domain ram count 1024 }
  resource B { rid 11 domain ram count 1024 }
  resource C { rid 12 domain hbm count 1024 }
  phase 0 {
    claim add { op add reads A B writes C count 1024 lane u stride unit }
  }
}
"""

_MAP = """
; resources then one operation
res A rid 10 n 1024
res B rid 11 n 1024
res C rid 12 n 1024 domain hbm
add C <- A, B n 1024 lane u stride unit
"""


def test_rop_frontend_emits_verified_claims():
    m = parse_rop_program(_ROP)
    assert {r.rid for r in m.resources.values()} == {10, 11, 12}
    claim = m.phases[0].claims[0]
    assert claim.op == "rop.add" and claim.rd == (10, 11) and claim.wr == (12,)
    assert claim.count == 1024
    assert is_legal(m)
    # text -> claims -> K_BCIR plan
    assert optimize(m, TargetProfile.x86_avx512(), Theta.cool()).by_claim()[1000].width == 16


def test_map_frontend_emits_verified_claims():
    m = parse_map(_MAP)
    assert {r.rid for r in m.resources.values()} == {10, 11, 12}
    claim = m.phases[0].claims[0]
    assert claim.op == "map.add" and claim.rd == (10, 11) and claim.wr == (12,)
    assert claim.count == 1024
    assert is_legal(m)
    assert optimize(m, TargetProfile.nvidia_ptx(), Theta.cool()).by_claim()[1000].width == 32
