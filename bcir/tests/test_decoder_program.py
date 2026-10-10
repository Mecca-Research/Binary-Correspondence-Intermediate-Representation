"""QUAL-1 and the oracle's half of QUAL-2: a decoder's generation as a BCIR program.

`decoder_program` states a greedy generation as one claim per decoder operation per token; the
program verifies under every R-law, K_BCIR plans it, GEM hydrates it, and the StreamPack it
hydrates to survives every projection -- the native wire, DER, OER and JER -- unchanged, as do
the module and the plan. `run_decoder_pack` then executes that pack, decoded from any of the
projections, and must produce `decode_with_kv_cache`'s tokens and logits bit for bit: the
program is the decoder, not a description of it. The C interpreter is held to the monolithic C
runner the same way in `test_decoder_gem.py`.
"""

from __future__ import annotations

from collections import Counter

from bcir.abi import encode
from bcir.abi import execution_plan_abi as plan_abi
from bcir.abi.streampack_abi import decode
from bcir.asn1 import execution_plan as plan_asn1
from bcir.asn1 import program as program_asn1
from bcir.asn1 import streampack as pack_asn1
from bcir.frontends.models.decode import (
    DecoderSpec,
    KVCache,
    decode_with_kv_cache,
    head_logits,
)
from bcir.frontends.models.decoder_gem import (
    DecoderProgramError,
    load_decoder_pack,
    pack_dispatch_order,
    run_decoder_pack,
)
from bcir.frontends.models.decoder_program import (
    MAX_LAYERS,
    OP_ARGMAX,
    OP_ATTENTION,
    OP_EMBED,
    OP_HEAD,
    OP_KV_APPEND,
    OP_MATVEC,
    OP_MATVEC_ADD,
    OP_RMSNORM,
    OP_ROPE,
    OP_SWIGLU,
    RID_CTX,
    RID_G_FINAL,
    RID_H,
    RID_Q,
    RID_X,
    claim_traffic,
    decoder_program,
    layer_rid,
    program_traffic,
    q8_bytes,
)
from bcir.gem.execution_plan import plan_from_realization
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.tests import decoder_fixtures as fx
from bcir.verify import verify_all

_PROJECTIONS = (
    ("native", lambda p: decode(encode(p))),
    ("der", lambda p: pack_asn1.decode_pack(pack_asn1.encode_pack(p))),
    ("oer", lambda p: pack_asn1.decode_pack_oer(pack_asn1.encode_pack_oer(p))),
    ("jer", lambda p: pack_asn1.decode_pack_jer(pack_asn1.encode_pack_jer(p))),
)


def _monolithic(spec, weights, prompt, max_new):
    """The oracle's monolithic generation: decode_with_kv_cache's ids, and the logits that
    chose the last of them (the cache re-run to the final row)."""
    ids = decode_with_kv_cache(prompt, spec, weights, max_new)
    cache, row = KVCache(spec), None
    for token in list(prompt) + ids[:-1]:
        row = cache._step_row(weights.embedding.row(token), spec, weights)
    return ids, head_logits(row, weights)


def test_a_program_states_every_operation_of_every_token_once():
    """The fixture's GQA decoder (two layers) over a three-token prompt and four new tokens: one
    embedding per position the decoder steps through, thirteen operations per layer per
    position, a final norm, head and argmax per generated token -- each its own claim, in a
    chain of phases where q/k/v and gate/up share theirs."""
    spec, _, _ = fx.model(tied=False)
    prog = fx.program(spec)
    positions, layers, new = prog.capacity - 1, spec.n_layers, fx.MAX_NEW
    claims = [c for ph in prog.module.phases for c in ph.claims]
    ops = Counter(c.op for c in claims)
    assert prog.claims == len(claims) == positions * (1 + 13 * layers) + 3 * new
    assert ops == {
        OP_EMBED: positions,
        OP_RMSNORM: positions * 2 * layers + new,
        OP_MATVEC: positions * 5 * layers,
        OP_ROPE: positions * layers,
        OP_KV_APPEND: positions * layers,
        OP_ATTENTION: positions * layers,
        OP_MATVEC_ADD: positions * 2 * layers,
        OP_SWIGLU: positions * layers,
        OP_HEAD: new,
        OP_ARGMAX: new,
    }
    assert [c.id for c in claims] == list(range(len(claims)))  # dense: the tables' indices
    phases = prog.module.phases
    assert all(ph.deps == ((ph.phase_id - 1,) if ph.phase_id else ()) for ph in phases)
    sizes = Counter(tuple(sorted(c.op for c in ph.claims)) for ph in phases)
    assert sizes[(OP_MATVEC,) * 3] == positions * layers  # q, k and v in one phase
    assert sizes[(OP_MATVEC,) * 2] == positions * layers  # gate and up in one phase
    # the tape: argmax claims write positions prompt_len .. capacity-1, embeds read 0 .. -2
    assert sorted(c.offset for c in claims if c.op == OP_ARGMAX) == list(
        range(len(fx.PROMPT), prog.capacity)
    )
    assert sorted(c.offset for c in claims if c.op == OP_EMBED) == list(range(positions))
    for c in claims:  # attention's work grows with the rows it attends
        if c.op == OP_ATTENTION:
            assert c.count == 2 * spec.d_model * (c.offset + 1)
    q0 = next(c for c in claims if c.op == OP_MATVEC and c.wr == (RID_Q,))
    assert q0.rd == (RID_H, layer_rid(0, 1)) and q0.count == spec.d_model * spec.d_model
    o0 = next(c for c in claims if c.op == OP_MATVEC_ADD and c.rd[0] == RID_CTX)
    assert o0.rd[2] == RID_X and o0.wr == (RID_X,)


def test_the_program_verifies_plans_and_hydrates_clean():
    """Every R-law holds over the program, its plan and its pack (no diagnostic), and the pack
    is the program: one segment and one block per claim, index-aligned, the block carrying the
    claim's offset and work."""
    spec, _, _ = fx.model(tied=False)
    prog = fx.program(spec)
    pack, result = fx.planned_pack(prog)
    h, theta = TargetProfile.x86_avx512(), Theta.cool()
    assert verify_all(prog.module, result=result, pack=pack, h=h, theta=theta) == []
    by_id = {c.id: c for ph in prog.module.phases for c in ph.claims}
    assert len(pack.segments) == len(pack.blocks) == prog.claims
    for seg, block in zip(pack.segments, pack.blocks):
        claim = by_id[seg.claim_id]
        assert (seg.opcode, tuple(seg.reads), tuple(seg.writes)) == (claim.op, claim.rd, claim.wr)
        assert (block.base, block.count) == (claim.offset, claim.count)


def test_the_module_the_plan_and_the_pack_survive_every_projection():
    """JSON <-> ASN.1 <-> GEM: the program's module through JER, its plan through the binary
    ABI, DER, OER and JER, and its pack through the native wire, DER, OER and JER -- each back
    to an equal value, each re-encoding to the same octets."""
    spec, _, _ = fx.model(tied=True)
    prog = fx.program(spec)
    pack, result = fx.planned_pack(prog)
    jer = program_asn1.module_to_jer(prog.module)
    module = program_asn1.jer_to_module(jer)
    assert module == prog.module and program_asn1.module_to_jer(module) == jer
    plan = plan_from_realization(prog.module, result, TargetProfile.x86_avx512(), "eft")
    for enc, dec in (
        (plan_abi.encode_plan, plan_abi.decode_plan),
        (plan_asn1.encode_plan_der, plan_asn1.decode_plan_der),
        (plan_asn1.encode_plan_oer, plan_asn1.decode_plan_oer),
        (plan_asn1.encode_plan_jer, plan_asn1.decode_plan_jer),
    ):
        octets = enc(plan)
        assert dec(octets) == plan and enc(dec(octets)) == octets
    native = encode(pack)
    for name, project in _PROJECTIONS:
        assert encode(project(pack)) == native, name


def test_the_oracle_runs_the_pack_as_the_monolithic_decoder_bit_for_bit():
    """The pack, decoded from each projection and executed in GEM order by the oracle's
    kernels, generates decode_with_kv_cache's tokens and the very logits that chose the last
    one -- tied and untied heads, grouped-query attention, two layers, and a model wide
    enough that a kernel rounding differently would show in the logits' last bits."""
    for tied, wide in ((False, False), (True, False), (False, True)):
        spec, weights, _ = fx.model(tied=tied, wide=wide)
        pack, _ = fx.planned_pack(fx.program(spec))
        ids, logits = _monolithic(spec, weights, fx.PROMPT, fx.MAX_NEW)
        for name, project in _PROJECTIONS:
            run = run_decoder_pack(project(pack), spec, weights, fx.PROMPT)
            assert run.tokens == ids, (tied, wide, name)
            assert run.logits == logits, (tied, wide, name)  # bit for bit, not approximately
            assert run.order == pack_dispatch_order(pack)
            assert sum(run.claims.values()) == len(pack.segments)


def test_the_program_is_its_own_tape():
    """The interpreter takes the tape's length and the prompt's from the program's argmax
    claims, and refuses a prompt of any other length or with an id outside the vocabulary."""
    spec, weights, _ = fx.model(tied=False)
    for prompt_len, max_new in ((1, 1), (3, 4), (5, 2)):
        pack, _ = fx.planned_pack(decoder_program(spec, prompt_len, max_new))
        _, capacity, plen = load_decoder_pack(pack, spec)
        assert (capacity, plen) == (prompt_len + max_new, prompt_len)
    pack, _ = fx.planned_pack(fx.program(spec))
    for prompt in ([1, 2], [1, 2, 3, 4], [1, 2, spec.vocab_size], [1, -1, 2]):
        try:
            run_decoder_pack(pack, spec, weights, prompt)
        except DecoderProgramError as exc:
            assert exc.kind == "arg", exc
        else:
            raise AssertionError(f"prompt {prompt} was accepted")


def _refused(pack, spec, weights, kind: str) -> None:
    try:
        run_decoder_pack(pack, spec, weights, fx.PROMPT)
    except DecoderProgramError as exc:
        assert exc.kind == kind, exc
    else:
        raise AssertionError(f"a {kind} violation ran")


def test_the_oracle_refuses_a_pack_that_is_not_this_models_program():
    """Each edit makes the pack a program this model cannot run, and the interpreter refuses it
    before anything runs ("program") or at the claim that breaks the data flow ("order"); the
    C interpreter refuses the same packs the same way (test_decoder_gem.py)."""
    spec, weights, _ = fx.model(tied=False)
    pack, _ = fx.planned_pack(fx.program(spec))
    rmsnorm = fx.claim_at(pack, OP_RMSNORM)
    q = fx.claim_at(pack, OP_MATVEC)
    attention = fx.claim_at(pack, OP_ATTENTION, 0)
    append = fx.claim_at(pack, OP_KV_APPEND, 0)
    program = (
        fx.edit_claim(pack, q, writes=(RID_X,)),  # q's projection written over the residual
        fx.edit_claim(pack, q, block={"count": 1}),  # work that is not the matrix's
        fx.edit_claim(pack, q, reads=(RID_H, layer_rid(spec.n_layers, 1))),  # a layer too many
        fx.edit_claim(pack, rmsnorm, opcode="dec.softmax"),  # an op the scheme does not name
        fx.edit_claim(pack, attention, block={"base": 1 << 40}),  # a position off the tape
        fx.drop_claim(pack, fx.claim_at(pack, OP_ARGMAX, 4)),  # a hole in the tape
        fx.edit_claim(pack, fx.claim_at(pack, OP_EMBED, 1), block={"base": 99}),  # off the tape
    )
    for edited in program:
        _refused(edited, spec, weights, "program")
    skipped = fx.drop_claim(pack, append)  # position 0's row neither appended nor attended:
    skipped = fx.drop_claim(skipped, fx.claim_at(skipped, OP_ATTENTION, 0))  # a hole at row 0
    order = (
        fx.swap_positions(pack, append, attention),  # attention before its row is appended
        skipped,  # position 1's append would leave row 0 of the cache unwritten
        fx.swap_positions(pack, fx.claim_at(pack, OP_EMBED, 0), fx.claim_at(pack, OP_EMBED, 1)),
        fx.drop_claim(pack, fx.claim_at(pack, OP_HEAD)),  # an argmax with no logits for it
        fx.drop_claim(pack, append),  # attention over a row never appended
    )
    bare = fx.layerless(pack, RID_G_FINAL)  # runs: a program needs no layer ...
    assert run_decoder_pack(bare, spec, weights, fx.PROMPT).tokens
    e0, e1 = fx.claim_at(bare, OP_EMBED, 0), fx.claim_at(bare, OP_EMBED, 1)
    order += (fx.swap_positions(bare, e0, e1),)  # ... but its embeddings run in order
    for edited in order:
        _refused(edited, spec, weights, "order")
    _refused(pack, DecoderSpec(**{**spec.__dict__, "activation": "gelu"}), weights, "program")


def test_a_run_reads_what_it_wrote_or_zeros():
    """A program that reads an activation before writing it (its first attention norm
    dropped) still runs -- the interpreter executes the program it is given -- and reads
    zeros, never a value another run left: two runs agree, and the C interpreter computes the
    same tokens (test_decoder_gem.py)."""
    spec, weights, _ = fx.model(tied=False)
    pack, _ = fx.planned_pack(fx.program(spec))
    holed = fx.drop_claim(pack, fx.claim_at(pack, OP_RMSNORM, 0))
    first = run_decoder_pack(holed, spec, weights, fx.PROMPT)
    second = run_decoder_pack(holed, spec, weights, fx.PROMPT)
    assert (first.tokens, first.logits) == (second.tokens, second.logits)
    assert first.logits != run_decoder_pack(pack, spec, weights, fx.PROMPT).logits


def final_norm_claim(pack) -> int:
    """The first final norm's claim (the rmsnorm that reads G_FINAL)."""
    return next(
        s.claim_id for s in pack.segments if s.opcode == OP_RMSNORM and s.reads[1] == RID_G_FINAL
    )


def test_final_is_a_resource_of_its_own():
    """FINAL and H are distinct resources: a program that writes H between the final norm and
    the head (a legal claim, the first attention norm run again there) generates exactly what
    the program without it does -- the head reads FINAL. The C interpreter, whose monolithic
    sibling keeps the final row in h, is held to the same (test_decoder_gem.py)."""
    spec, weights, _ = fx.model(tied=False)
    pack, _ = fx.planned_pack(fx.program(spec))
    base = run_decoder_pack(pack, spec, weights, fx.PROMPT)
    extra = fx.insert_copy(pack, fx.claim_at(pack, OP_RMSNORM), after=final_norm_claim(pack))
    run = run_decoder_pack(extra, spec, weights, fx.PROMPT)
    assert (run.tokens, run.logits) == (base.tokens, base.logits)
    assert len(run.order) == len(base.order) + 1


def test_decoder_program_refuses_what_it_cannot_lower():
    spec, _, _ = fx.model(tied=False)
    gelu = DecoderSpec(**{**spec.__dict__, "activation": "gelu"})
    deep = DecoderSpec(**{**spec.__dict__, "n_layers": MAX_LAYERS + 1})
    for args in ((gelu, 1, 1), (deep, 1, 1), (spec, 0, 1), (spec, 1, 0), (spec, 1 << 20, 1)):
        try:
            decoder_program(*args)
        except ValueError:
            continue
        raise AssertionError(f"decoder_program{args[1:]} lowered")
    for bad in ("spec", None):
        try:
            decoder_program(bad, 1, 1)
        except ValueError:
            continue
        raise AssertionError("a non-spec lowered")
    for prompt_len, max_new in ((True, 1), (1, 1.0)):
        try:
            decoder_program(spec, prompt_len, max_new)
        except ValueError:
            continue
        raise AssertionError("a non-integer length lowered")


def test_planned_traffic_is_each_claims_operands():
    """The traffic model the C interpreter's counters are held to (test_decoder_gem.py):
    BCIRQ8 weights at a code an element and two exponent octets a group, activations in
    double, the tape in int32 -- summed per op over every claim."""
    spec, _, _ = fx.model(tied=False)
    prog = fx.program(spec)
    d, kvd, ff, vocab = spec.d_model, spec.kv_dim, spec.d_ff, spec.vocab_size
    assert q8_bytes(32) == 34 and q8_bytes(33) == 37 and q8_bytes(8) == 10
    claims = [c for ph in prog.module.phases for c in ph.claims]
    q = next(c for c in claims if c.op == OP_MATVEC)
    assert claim_traffic(spec, q) == (8 * d + q8_bytes(d * d), 8 * d)
    down = next(c for c in claims if c.op == OP_MATVEC_ADD and c.count == ff * d)
    assert claim_traffic(spec, down) == (8 * ff + q8_bytes(ff * d) + 8 * d, 8 * d)
    att = next(c for c in claims if c.op == OP_ATTENTION and c.offset == 2)
    assert claim_traffic(spec, att) == (8 * d + 2 * 3 * kvd * 8, 8 * d)
    head = next(c for c in claims if c.op == OP_HEAD)
    assert claim_traffic(spec, head) == (8 * d + q8_bytes(vocab * d), 8 * vocab)
    totals = program_traffic(prog)
    for op in {c.op for c in claims}:
        assert totals[op] == tuple(
            map(sum, zip(*(claim_traffic(spec, c) for c in claims if c.op == op)))
        )
    assert totals["total"] == tuple(map(sum, zip(*(v for k, v in totals.items() if k != "total"))))
