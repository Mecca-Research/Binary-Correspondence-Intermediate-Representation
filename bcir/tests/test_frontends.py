"""CT3: ROP (declarative) + MAP (macro-assembly) front-end tests."""

import threading
from concurrent.futures import ThreadPoolExecutor

from bcir.frontends import parse_map, parse_rop_program
from bcir.frontends.map import MapError
from bcir.etl.parse import ParseError
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


def test_rop_resource_resolution_is_per_parse_not_process_global():
    import bcir.frontends.rop as rop

    second = (_ROP.replace("rid 10 domain", "rid 110 domain")
              .replace("rid 11 domain", "rid 111 domain")
              .replace("rid 12 domain", "rid 112 domain"))
    rendezvous = threading.Barrier(2)
    original = rop._P.parse

    def synchronized_parse(self):
        rendezvous.wait(timeout=5)
        return original(self)

    rop._P.parse = synchronized_parse
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_future = pool.submit(rop.parse_rop_program, _ROP)
            second_future = pool.submit(rop.parse_rop_program, second)
            first = first_future.result(timeout=10)
            other = second_future.result(timeout=10)
    finally:
        rop._P.parse = original
    assert first.phases[0].claims[0].rd == (10, 11)
    assert other.phases[0].claims[0].rd == (110, 111)


def test_declarative_frontends_reject_ambiguous_or_silent_defaults():
    bad_map = (
        "res A rid 10\nres A rid 11\n",
        "res A rid 10\nres B rid 10\n",
        "res A rid 10\nres C rid 12\nadd C <- A, MISSING\n",
        "res A rid 10\nres C rid 12\nadd C <- A lane telepathy\n",
        "res A rid 10 rid 11\n",
        "res A rid\n",
    )
    for source in bad_map:
        try:
            parse_map(source)
            raise AssertionError(f"MAP must reject ambiguous input: {source!r}")
        except MapError:
            pass

    bad_rop = (
        _ROP.replace("op add", "op telepathy"),
        _ROP.replace("reads A B", "reads A MISSING"),
        _ROP.replace("phase 0 {", "phase 0 {", 1).replace(
            "\n  }\n}", "\n  }\n  phase 0 { }\n}", 1
        ),
        _ROP + " trailing",
        _ROP.replace("count 1024 lane", "count 1024 count 3 lane"),
    )
    for source in bad_rop:
        try:
            parse_rop_program(source)
            raise AssertionError("ROP must reject ambiguous input")
        except ParseError:
            pass
