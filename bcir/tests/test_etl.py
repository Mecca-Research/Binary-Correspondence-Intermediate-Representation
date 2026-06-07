"""Event Transduction Layer (M5) tests: text -> claims, binary -> records, FSM."""

from bcir.etl import Lexer, State, Transducer, Transition, decode, parse_rop
from bcir.etl.binary import NVME_SQE_HEADER
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.model import Lane, StrideClass
from bcir.verify import is_legal

_SRC = "claim add { reads A, B; writes C; count 1024; lane u; stride unit }"


def test_lexer_tokenizes_and_skips_ws():
    kinds = [t.kind for t in Lexer().tokenize("claim add { count 16 ; }")]
    assert kinds == ["IDENT", "IDENT", "LBRACE", "IDENT", "INT", "SEMI", "RBRACE", "EOF"]


def test_parse_rop_emits_bcir_claims():
    m = parse_rop(_SRC)
    assert {r.name for r in m.resources.values()} == {"A", "B", "C"}
    claim = m.phases[0].claims[0]
    assert claim.op == "rop.add"
    assert claim.rd == (10, 11) and claim.wr == (12,)
    assert claim.count == 1024
    assert claim.lane == Lane.U and claim.stride_class == StrideClass.UNIT


def test_parsed_module_verifies_and_plans():
    # Close the loop: text -> claims -> K_BCIR plan (vec16 on AVX-512 cool).
    m = parse_rop(_SRC)
    assert is_legal(m)
    res = optimize(m, TargetProfile.x86_avx512(), Theta.cool())
    assert res.by_claim()[1000].width == 16


def test_binary_decode_nvme_sqe_header():
    # opcode=0x01, command_id=0x1234, namespace_id=0xAABBCCDD (little-endian, byte 1 reserved).
    data = bytes([0x01, 0x00, 0x34, 0x12, 0xDD, 0xCC, 0xBB, 0xAA])
    fields = decode(NVME_SQE_HEADER, data, endianness="little")
    assert fields["opcode"] == 0x01
    assert fields["command_id"] == 0x1234
    assert fields["namespace_id"] == 0xAABBCCDD


def test_transducer_accepts_and_rejects():
    states = [State("s0"), State("s1"), State("s_acc", accepting=True)]
    tx = [Transition("s0", "s1", on="claim", action="begin"),
          Transition("s1", "s_acc", on="ident", action="capture_name")]
    t = Transducer(states, tx, start="s0")
    ok = t.run(["claim", "ident"])
    assert ok.accepted and ok.final_state == "s_acc"
    assert ("capture_name", "ident") in ok.captures
    assert not t.run(["claim"]).accepted          # stops in non-accepting s1
    assert not t.run(["ident"]).accepted          # no edge from s0 on 'ident'
