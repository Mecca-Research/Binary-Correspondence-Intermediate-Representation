"""Event Transduction Layer (M5) tests: text -> claims, binary -> records, FSM."""

from bcir.etl import Lexer, State, Transducer, Transition, decode, parse_rop
from bcir.etl.parse import ParseError
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
    tx = [
        Transition("s0", "s1", on="claim", action="begin"),
        Transition("s1", "s_acc", on="ident", action="capture_name"),
    ]
    t = Transducer(states, tx, start="s0")
    ok = t.run(["claim", "ident"])
    assert ok.accepted and ok.final_state == "s_acc"
    assert ("capture_name", "ident") in ok.captures
    assert not t.run(["claim"]).accepted  # stops in non-accepting s1
    assert not t.run(["ident"]).accepted  # no edge from s0 on 'ident'


def test_parse_rop_refuses_with_a_parse_error_every_time():
    """PORT-4: each refusal of the mini-ROP grammar is a ParseError at the token it is about -- an unknown lane or
    stride had raised a KeyError (a traceback, not a verdict), a count past the signed 64-bit wire domain had been
    taken, a second `reads` had replaced the first and kept its resources, read by no claim, and an unknown statement
    was placed at the token after it. The C twin (`runtime/c/bcir_etl.c`) refuses each at the same offset."""
    for src, words in (
        ("claim a { lane q }", "unknown lane 'q' at offset 15"),
        ("claim a { stride zz }", "unknown stride 'zz' at offset 17"),
        (
            "claim a { count 9223372036854775808 }",
            "count 9223372036854775808 is past the signed 64-bit wire domain at offset 16",
        ),
        ("claim a { reads A; reads B }", "repeated statement 'reads' at offset 19"),
        ("claim a { foo X }", "unknown statement 'foo' at offset 10"),
    ):
        try:
            parse_rop(src)
        except ParseError as e:
            assert str(e) == words, (src, str(e))
        else:
            raise AssertionError(f"{src!r} parsed")
    assert parse_rop("claim a { count 9223372036854775807 }").phases[0].claims[0].count == 2**63 - 1
    assert parse_rop("claim a { lane GGG; stride Random }").phases[0].claims[0].lane == Lane.GGG
