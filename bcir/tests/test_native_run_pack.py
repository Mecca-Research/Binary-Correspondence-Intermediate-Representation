"""PX-2b (PACK): the native training step run from a StreamPack and its ExecutionPlanV1.

`native_pack.training_pack` states the program PX-2a runs as BCIR's own artifacts -- a module of its
calls and values, planned by K_BCIR, hydrated by GEM into a StreamPack, placed by an ExecutionPlanV1
-- and `bcir_decoder_lower_pack` takes the two and the state and nothing else: their verifiers (the
plan's alias law among them), then each claim GEM's executor dispatches lowered to the record that
runs it. The witness holds every pack of the trace corpus's program runs to its program's words and
its run to the C step on the same case, bit for bit; the others hold another legal placement to the
same bits, every refusal to its reason with nothing written, the module and the plan to the R-laws
and the plan's laws, the builder to the values it can state, and the vocabulary and the reasons to
the C's own tables.
"""

import copy
import ctypes as ct
import dataclasses
import re
import tempfile
import zlib
from array import array
from collections import Counter
from pathlib import Path

from bcir.abi.execution_plan_abi import encode_plan
from bcir.abi.streampack_abi import encode
from bcir.hosted.models import native as nat
from bcir.hosted.models.native_launch import RECORD_WORDS, _layout
from bcir.hosted.models.native_pack import (
    OPS,
    REASONS,
    Placement,
    pack_artifacts,
    placed_words,
    program_pack,
    training_pack,
)
from bcir.hosted.models.native_program import training_step_program
from bcir.tests import native_decoder_fixtures as nf
from bcir.tests.plan_fixtures import raw_encode

#: The host's C compiler, or None: the witnesses' one capability (a fault sweep `--require`s it,
#: so a host without one is never read as a sweep the witnesses passed).
_CC = __import__("bcir.toolchain", fromlist=["host_c_compiler"]).host_c_compiler()
_C = Path(__file__).resolve().parents[2] / "runtime" / "c"
_SENTINEL = 0xA5A5A5A5A5A5A5A5


def test_a_pack_lowers_to_its_program_and_trains_the_c_steps_bits():
    """Every pack run of the trace corpus -- every case from the program of the C step's own
    decisions, its state cases under each of the planner's other choices -- lowers to exactly its
    program's words and trains the C step's state and events on the same case bit for bit
    (`train.plan.pack.mismatch` 0), and the packs name every claim the vocabulary has but none
    outside it."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.pack_library(Path(tmp) / "pack")
        if path is None:
            return  # no C compiler, or no C units (an installed package): a documented skip
        problems, names, runs = [], Counter(), 0
        for case, choices, step, run in nf.pack_runs(path):
            runs += 1
            found = nf.pack_problems(step, run)
            if found:
                problems.append((case, choices, found[:2]))
            for phase in run.pack.module.phases:
                names.update(claim.op for claim in phase.claims)
        assert not problems, problems[:3]
        assert runs == nf.program_run_count(), (runs, nf.program_run_count())
        assert set(OPS) <= set(names), sorted(set(OPS) - set(names))
        gemms = {name for name in names if name.startswith("train.gemm.")}
        assert set(names) - set(OPS) == gemms, sorted(set(names) - set(OPS) - gemms)
        # every role, every orientation, accumulation and both panels
        for part in (".alone.", ".pair.", ".chain.", ".rest.", ".nt", ".nn", ".tn", ".acc", ".pa",
                     ".pb"):  # fmt: skip
            assert any(part in name for name in gemms), (part, sorted(gemms))


def _moved(tp, recompute: int) -> list[Placement]:
    """`tp`'s placements with every activation and scratch buffer's span laid out in reverse order
    within its arena -- the values inside a buffer at their offsets within it -- so every
    transient value moves and no two values of different buffers meet: a legal placement other
    than the C step's."""
    lay = _layout(training_step_program(tp.spec, tp.batch, tp.time, recompute))
    base: dict[tuple[int, int], int] = {}
    for space in (5, 6):
        spans = sorted((off, size) for sp, off, size in lay.values() if sp == space)
        at = 0
        for off, size in reversed(spans):
            base[space, off] = at
            at += size
    out = []
    for p in tp.placements:
        space = {"activations": 5, "scratch": 6}.get(p.bank)
        if space is None:
            out.append(p)
            continue
        lo = max(off for sp, off in base if sp == space and off <= p.offset)
        out.append(dataclasses.replace(p, offset=base[space, lo] + p.offset - lo))
    return out


def test_the_placement_is_the_plans():
    """A plan that places every activation and scratch value elsewhere -- the buffers in reverse
    order, a legal placement the C step never uses -- lowers to the same program at the new
    places and trains the C step's state and events bit for bit: the operands' places are the
    plan's, and the bits do not depend on them."""
    cases = [c for c in nf.trace_corpus() if c.steps == 2 and c.recompute in (0, 1)][:2]
    spec, batch, time_ = nf.PANELS_BENCH
    cases.append(nf.TraceCase(spec, batch, time_, "bf16", 0b10, "need", 1))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.pack_library(Path(tmp) / "pack")
        if path is None:
            return
        for case in cases:
            step = nf.trace_case(path, case, trace=False)
            trainer, decisions = nf._case_trainer(path, case)
            trainer.close()
            tp = training_pack(case.spec, case.batch, case.time, **decisions)
            moved = _moved(tp, decisions["recompute"])
            assert moved != list(tp.placements)
            run = nf.pack_case(path, case, placements=moved)
            assert run.status == 0, (case, run.status, run.why)
            assert run.words != run.pack.words, case  # the operands moved
            differ = {i % RECORD_WORDS for i, (a, b) in enumerate(zip(run.words, run.pack.words))
                      if a != b}  # fmt: skip
            assert differ <= {12 + 3 * q for q in range(10)}, (case, sorted(differ))  # offsets
            # exactly where the plan put them (PX-4's relocation oracle)
            assert run.words == placed_words(tp, moved), case
            assert placed_words(tp, tp.placements) == tp.words, case
            assert (run.state, run.events) == (step.state, step.events), case


def _lower(trainer, pack: bytes, plan: bytes, *, work_short=0, capacity_short=0):
    """`bcir_decoder_lower_pack` called directly into a program buffer of sentinel words: (status,
    why, records, whether the program buffer is untouched, the words). A size figure past any of
    these packs' (a gigabyte of work, 2**24 records) is status -1, never lent."""
    lib = trainer.lib
    sizes = lib.bcir_decoder_pack_sizes
    sizes.argtypes = [ct.c_char_p, ct.c_size_t, ct.c_char_p, ct.c_size_t,
                      ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_int)]  # fmt: skip
    sizes.restype = ct.c_int
    records, work_bytes, why = ct.c_size_t(), ct.c_size_t(), ct.c_int()
    rc = sizes(pack, len(pack), plan, len(plan), ct.byref(records), ct.byref(work_bytes),
               ct.byref(why))  # fmt: skip
    if rc:
        return rc, why.value, 0, True, None
    if work_bytes.value > 1 << 30 or records.value > 1 << 24:
        return -1, 0, 0, True, None  # a figure no pack of these tests needs: never allocated
    work = (ct.c_uint64 * (work_bytes.value // 8 + 1))()
    capacity = records.value - capacity_short
    program = (ct.c_uint64 * (RECORD_WORDS * max(records.value, 1)))(
        *([_SENTINEL] * (RECORD_WORDS * max(records.value, 1)))
    )
    fn = lib.bcir_decoder_lower_pack
    fn.argtypes = [ct.POINTER(nat._State), ct.c_char_p, ct.c_size_t, ct.c_char_p, ct.c_size_t,
                   ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_uint64), ct.c_size_t,
                   ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_int)]  # fmt: skip
    fn.restype = ct.c_int
    done = ct.c_size_t(77)
    rc = fn(ct.byref(trainer.state), pack, len(pack), plan, len(plan), work,
            work_bytes.value - work_short, program, capacity, ct.byref(done), ct.byref(why))  # fmt: skip
    words = list(program)
    return rc, why.value, done.value, all(w == _SENTINEL for w in words), words


def _segment(pack, op_prefix: str, nth: int = 0) -> int:
    hits = [i for i, seg in enumerate(pack.segments) if seg.opcode.startswith(op_prefix)]
    return hits[nth]


def _edit_pack(pack, i, block=None, **segment):
    p = dataclasses.replace(pack, segments=list(pack.segments), blocks=list(pack.blocks),
                            trace_notes=list(pack.trace_notes))  # fmt: skip
    if segment:
        p.segments[i] = dataclasses.replace(p.segments[i], **segment)
    if block:
        p.blocks[i] = dataclasses.replace(p.blocks[i], **block)
    return p


def _edit_lifetimes(plan, edit):
    """A copy of `plan` with its lifetimes passed through `edit` (a list in, a list out)."""
    p = copy.deepcopy(plan)
    p.lifetimes = sorted(edit(list(p.lifetimes)), key=lambda lt: lt.rid)
    return p


def _refusals(tp, pack, plan):
    """(name, pack bytes, plan bytes, the reason the lowering must give) for each class of pack
    or plan the lowering refuses, on `tp`'s artifacts (a v1 program of the small spec). The plans
    are written by `plan_fixtures.raw_encode`, without the wire laws: the public encoder refuses
    to write the plans the C must refuse (two live values on one span among them)."""
    names = {r: k for k, r in REASONS.items()}
    claims = [c for ph in tp.module.phases for c in ph.claims]
    by_id = {c.id: c for c in claims}
    place = {p.rid: p for p in tp.placements}
    transient = [p for p in tp.placements if p.bank in ("activations", "scratch")]
    spans = [p for p in transient if p.last > p.first]
    gemm = _segment(pack, "train.gemm.")
    # the rotation, in place (not the table's fill, ROPE-TABLE, which shares its prefix)
    rope = next(i for i, seg in enumerate(pack.segments) if seg.opcode == "train.rope")
    embed = _segment(pack, "train.embed")
    lt = {x.rid: x for x in plan.lifetimes}
    # two activation values live at once: the first RMS norm's input and output
    rms = by_id[pack.segments[_segment(pack, "train.rms")].claim_id]
    x0, n1 = rms.rd[0], rms.wr[0]
    tokens = by_id[pack.segments[embed].claim_id].rd[0]
    targets = 4 * tp.tensors + 3  # module docstring: after the parameters and the workspace
    assert place[targets].bank == "targets", place[targets]
    end = max(p.offset + p.size for p in tp.placements if p.bank == "activations")
    other = next(r for r in by_id[pack.segments[gemm].claim_id].rd)
    # two weight tensors of one size, apart: swapped, each still whole and alone, but the arena's
    # run of tensors in RID order is no longer one span
    weights = sorted((p.rid, p.size) for p in tp.placements if p.bank == "weights")
    swap = next((r, q) for i, (r, size) in enumerate(weights) for q, other in weights[i + 2 :]
                if other == size)  # fmt: skip
    a = by_id[pack.segments[gemm].claim_id].rd[0]
    out = [
        ("op", encode(_edit_pack(pack, gemm, opcode="train.gemm.alone.xx")), raw_encode(plan),
         "op"),
        ("op-order", encode(_edit_pack(pack, gemm, opcode="train.gemm.alone.nt.pa.acc")),
         raw_encode(plan), "op"),
        ("rid", encode(pack), raw_encode(_edit_lifetimes(
            plan, lambda ls: [x for x in ls if x.rid != spans[0].rid])), "rid"),
        ("late", encode(pack), raw_encode(_edit_lifetimes(plan, lambda ls: [
            dataclasses.replace(x, first_phase=x.first_phase + 1, first_tick=0, last_tick=0)
            if x.rid == spans[0].rid else x for x in ls])), "live"),
        ("early", encode(pack), raw_encode(_edit_lifetimes(plan, lambda ls: [
            dataclasses.replace(x, last_phase=x.last_phase - 1, first_tick=0, last_tick=0)
            if x.rid == spans[0].rid else x for x in ls])), "live"),
        ("alias", encode(pack), raw_encode(_edit_lifetimes(plan, lambda ls: [
            dataclasses.replace(x, offset=lt[x0].offset) if x.rid == n1 else x for x in ls])),
         "plan"),
        ("bank", encode(pack), raw_encode(_edit_lifetimes(plan, lambda ls: [
            dataclasses.replace(x, bank="vram") if x.rid == spans[0].rid else x for x in ls])),
         "bank"),
        ("tokens-bank", encode(pack), raw_encode(_edit_lifetimes(plan, lambda ls: [
            dataclasses.replace(x, bank="activations", offset=end) if x.rid == tokens else x
            for x in ls])), "bank"),
        ("targets-bank", encode(pack), raw_encode(_edit_lifetimes(plan, lambda ls: [
            dataclasses.replace(x, bank="activations", offset=end) if x.rid == targets else x
            for x in ls])), "bank"),
        ("plain-bank", encode(pack), raw_encode(_edit_lifetimes(plan, lambda ls: [
            dataclasses.replace(x, bank="workspace") if x.rid == spans[0].rid else x
            for x in ls])), "bank"),
        ("extra-read", encode(_edit_pack(pack, gemm, reads=pack.segments[gemm].reads + (other,))),
         raw_encode(plan), "slots"),
        ("short-read", encode(_edit_pack(pack, gemm, reads=pack.segments[gemm].reads[:-1])),
         raw_encode(plan), "slots"),
        ("split-in-place", encode(_edit_pack(pack, rope, writes=(x0,))), raw_encode(plan),
         "slots"),
        ("run", encode(pack), raw_encode(_edit_lifetimes(plan, lambda ls: [
            dataclasses.replace(x, offset=lt[swap[1]].offset) if x.rid == swap[0] else
            dataclasses.replace(x, offset=lt[swap[0]].offset) if x.rid == swap[1] else x
            for x in ls])), "run"),
        ("shape", encode(pack), raw_encode(_edit_lifetimes(plan, lambda ls: [
            dataclasses.replace(x, size_bytes=x.size_bytes - 4) if x.rid == a else x
            for x in ls])), "shape"),
        ("count", encode(_edit_pack(pack, gemm, block={"count": pack.blocks[gemm].count + 1})),
         raw_encode(plan), "block"),
        ("base", encode(_edit_pack(pack, embed, block={"base": 1})), raw_encode(plan), "block"),
        ("crc", encode(pack)[:-1] + bytes([encode(pack)[-1] ^ 1]), raw_encode(plan), "crc"),
    ]  # fmt: skip
    # a claim id outside 0..n-1, renumbered in the pack and the plan alike
    last = max(c.id for c in claims)
    p = dataclasses.replace(pack, segments=list(pack.segments), trace_notes=list(pack.trace_notes))
    i = next(k for k, seg in enumerate(p.segments) if seg.claim_id == last)
    p.segments[i] = dataclasses.replace(p.segments[i], claim_id=last + 7)
    p.trace_notes = [dataclasses.replace(t, claim_id=last + 7) if t.claim_id == last else t
                     for t in p.trace_notes]  # fmt: skip
    q = copy.deepcopy(plan)
    q.steps = [dataclasses.replace(s, claim_id=last + 7) if s.claim_id == last else s
               for s in q.steps]  # fmt: skip
    out.append(("claims", encode(p), raw_encode(q), "claims"))
    # the plan of another realization: one step more than the pack has segments
    q = copy.deepcopy(plan)
    q.steps = q.steps + [dataclasses.replace(q.steps[-1], claim_id=last + 1)]
    out.append(("binding", encode(pack), raw_encode(q), "provenance"))
    # a plan placed under another generation of one resource
    q = copy.deepcopy(plan)
    q.generations = [dataclasses.replace(g, data_gen=1) if g.rid == place[x0].rid else g
                     for g in q.generations]  # fmt: skip
    out.append(("stale", encode(pack), raw_encode(q), "stale"))
    # lifetimes in the schedule's ticks, not the phases'
    q = copy.deepcopy(plan)
    q.liveness = "schedule"
    out.append(("liveness", encode(pack), raw_encode(q), "liveness"))
    # PX-3b-2: a step's measured realization naming no alternative -- two bits, no workers, a
    # leading zero, too many workers, a non-ASCII digit, another letter, none at all
    at = next(i for i, st in enumerate(plan.steps) if st.claim_id == pack.segments[gemm].claim_id)
    for alt in ("i3", "t0", "t01", "t32768", "t\u0661", "w1", ""):
        q = copy.deepcopy(plan)
        q.steps[at] = dataclasses.replace(q.steps[at], candidate=q.steps[at].candidate + "@" + alt)
        out.append((f"alternative-{alt!a}", encode(pack), raw_encode(q), "alternative"))
    # two defects: the tokens' lifetime (ahead of every transient's) in a bank the state has not,
    # and the first norm's output on its input's bytes -- the plan's own laws refuse it before any
    # reason a lifetime gives the lowering (one walk hands it the lifetimes as it verifies them)
    out.append(("law-before-bank", encode(pack), raw_encode(_edit_lifetimes(plan, lambda ls: [
        dataclasses.replace(x, bank="vram") if x.rid == tokens else
        dataclasses.replace(x, offset=lt[x0].offset) if x.rid == n1 else x for x in ls])),
        "plan"))  # fmt: skip
    return [(name, pk, pl, names[reason]) for name, pk, pl, reason in out]


def test_a_refused_pack_lowers_nothing():
    """Every class of pack or plan the lowering refuses -- a name outside the vocabulary, a value
    no lifetime places, a claim outside its values' lifetimes, two live values on one span (the
    alias law), a bank the state has not or a slot does not take (the tokens', the targets', a
    plain operand's, the bf16 workspace's), operands not the claim's, a run of tensors not
    contiguous, a size no shape divides, a block not the claim's, claim ids out of range, a plan
    not the pack's or of another generation, lifetimes not in phases, a step's measured realization
    naming no alternative (PX-3b-2), a corrupt pack -- is refused
    with its reason (BCIR_DT_INVALID) and nothing written: the program buffer as it was, no record
    counted. Short buffers are BCIR_DT_CAPACITY; a pack for another state's contract is refused;
    and the intact pairs lower (the v1 trainer's and the bf16 trainer's own)."""
    case = nf.trace_corpus()[0]  # v1, the small spec, one step
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.pack_library(Path(tmp) / "pack")
        if path is None:
            return
        trainer, decisions = nf._case_trainer(path, case)
        try:
            tp = training_pack(case.spec, case.batch, case.time, **decisions)
            pack_bytes, plan_bytes, pack, plan = pack_artifacts(tp)
            assert raw_encode(plan) == plan_bytes  # the near misses differ in their edit alone
            rc, why, done, untouched, words = _lower(trainer, pack_bytes, plan_bytes)
            assert (rc, why, done) == (0, 0, tp.claims) and words[: len(tp.words)] == list(
                tp.words
            ), (rc, why, done)
            seen = set()
            for name, pk, pl, reason in _refusals(tp, pack, plan):
                rc, why, done, untouched, _ = _lower(trainer, pk, pl)
                assert (rc, why) == (1, reason), (name, rc, why, REASONS.get(why), reason)
                assert done == 0 and untouched, (name, done)
                seen.add(reason)
            # every reason the lowering has but CLAIMS' unreachable halves
            assert {r for r in REASONS if r > 100} <= seen, sorted(set(REASONS) - seen)
            # a plan header declaring more lifetimes than its bytes can hold, resealed: refused by
            # the sizes before any figure is lent (TRUNCATED), never a work size to allocate
            body = bytearray(plan_bytes[:-4])
            body[24:28] = (0x7FFFFFFF).to_bytes(4, "little")
            sealed = bytes(body) + (zlib.crc32(body) & 0xFFFFFFFF).to_bytes(4, "little")
            rc, why, done, untouched, _ = _lower(trainer, pack_bytes, sealed)
            assert (rc, why, done, untouched) == (1, 1, 0, True), (rc, why)
            for short in ({"work_short": 8}, {"capacity_short": 1}):
                rc, why, done, untouched, _ = _lower(trainer, pack_bytes, plan_bytes, **short)
                assert (rc, done, untouched) == (2, 0, True), (short, rc, why)
        finally:
            trainer.close()
        # the v1 pack on a bf16 trainer: its GEMMs take the workspace the pack does not name
        bf16 = next(c for c in nf.trace_corpus() if c.contract.startswith("bf16") and
                    c.spec == case.spec and c.steps == 1 and not c.recompute)  # fmt: skip
        trainer, decisions = nf._case_trainer(path, bf16)
        try:
            rc, why, done, untouched, _ = _lower(trainer, pack_bytes, plan_bytes)
            assert (rc, done, untouched) == (1, 0, True) and why > 100, (rc, why)
            # its own pack lowers; with the workspace placed in the activations it does not (the
            # workspace slot takes the workspace's bank alone)
            own = training_pack(bf16.spec, bf16.batch, bf16.time, **decisions)
            own_pack, own_plan, _, own_obj = pack_artifacts(own)
            rc, why, done, _, words = _lower(trainer, own_pack, own_plan)
            assert (rc, why, done) == (0, 0, own.claims), (rc, why, done)
            assert words[: len(own.words)] == list(own.words)
            ws = 4 * own.tensors + 1
            assert {p.rid: p.bank for p in own.placements}[ws] == "workspace"
            end = max(p.offset + p.size for p in own.placements if p.bank == "activations")
            moved = raw_encode(_edit_lifetimes(own_obj, lambda ls: [
                dataclasses.replace(x, bank="activations", offset=end) if x.rid == ws else x
                for x in ls]))  # fmt: skip
            rc, why, done, untouched, _ = _lower(trainer, own_pack, moved)
            assert (rc, REASONS.get(why), done, untouched) == (1, "bank", 0, True), (rc, why)
        finally:
            trainer.close()


def test_the_module_is_lawful_and_states_only_live_reads():
    """The module of a pack verifies under every R-law, its plan under the plan's laws against the
    module, the target and the pack (the lifetimes cover every touching phase, no two of a bank
    alias), what lives through the program -- the parameters, the workspace, the tokens and the
    targets -- lives through all of it, and a value of the C step's reuse shares its bytes with
    another only where their lifetimes are apart. A program that reads a span no live value holds
    -- an RMS norm's output read before the norm, or never written, or a value read after another
    overwrote its bytes -- is refused by the builder: it cannot state it."""
    from bcir.kbcir.cost import TargetProfile
    from bcir.verify import verify, verify_execution_plan

    for case in nf.trace_corpus()[:6]:
        decisions = dict(contract=nat._CONTRACTS[case.contract], recompute=case.recompute,
                         workspace_bytes=1 << 20 if "bf16" in case.contract else 0,
                         steps=case.steps, accumulation=case.accumulation)  # fmt: skip
        tp = training_pack(case.spec, case.batch, case.time, **decisions)
        _, _, pack, plan = pack_artifacts(tp)
        assert verify(tp.module) == [], case
        assert verify_execution_plan(tp.module, plan, target=TargetProfile.x86_avx512(),
                                     pack=pack) == [], case  # fmt: skip
        whole = (0, len(tp.program) - 1)
        for p in tp.placements:
            if p.bank not in ("activations", "scratch"):
                assert (p.first, p.last) == whole, (case, p)
        shared = 0
        rows = sorted(tp.placements, key=lambda p: (p.bank, p.offset))
        for i, p in enumerate(rows):
            for q in rows[i + 1 :]:
                if q.bank != p.bank or q.offset >= p.offset + p.size:
                    break
                shared += 1
                assert p.last < q.first or q.last < p.first, (case, p, q)
        assert shared, case  # the reuse is stated: values share bytes, at disjoint times
    from bcir.hosted.models.native_launch import ATTN_VJP, RMS, RMS_VJP, step_program

    case = nf.trace_corpus()[0]
    decisions = dict(contract=0, steps=1, accumulation=1)

    program = step_program(case.spec, case.batch, case.time, **decisions)
    first = next(i for i, launch in enumerate(program) if launch.records[0].kind == RMS)
    swapped = program[:first] + [program[first + 1], program[first]] + program[first + 2 :]
    # an RMS VJP's 1/rms (the head of dq) written between the attention VJP that writes dq and the
    # RoPE that reads it: dq's value is gone when it is read
    attn = next(i for i, launch in enumerate(program) if launch.records[0].kind == ATTN_VJP)
    norm = next(launch for launch in program if launch.records[0].kind == RMS_VJP)
    clobbered = program[: attn + 1] + [norm] + program[attn + 1 :]
    for edited in (swapped, program[:first] + program[first + 1 :], clobbered):
        try:
            program_pack(case.spec, case.batch, case.time, edited)
        except ValueError as refused:
            assert "no live value holds" in str(refused), refused
        else:
            raise AssertionError("a read of a span no live value holds was stated")


def test_the_vocabulary_and_the_reasons_are_the_cs():
    """The claims' names are the C's own table (bcir_decoder_pack.c's dp_names, in order), its GEMM
    grammar's parts are the model's, and the refusal reasons are the header's BCIR_DP_* and
    bcir_runtime.h's bcir_status -- read from their text, so neither rail can rename one alone."""
    unit = (_C / "bcir_decoder_pack.c").read_text(encoding="utf-8")
    table = re.search(r"dp_names\[OP_GEMM\] = \{(.*?)\};", unit, re.S)
    assert table, "bcir_decoder_pack.c has no dp_names table"
    assert tuple(re.findall(r'"([^"]+)"', table.group(1))) == OPS
    for part in ('".alone", ".pair", ".chain", ".rest"', '".nt", ".nn", ".tn"', '"", ".acc"',
                 '"", ".pa", ".pb"'):  # fmt: skip
        assert part in unit, part
    header = (_C / "bcir_decoder_pack.h").read_text(encoding="utf-8")
    enum = re.search(r"enum \{\s*BCIR_DP_OP = 101,(.*?)\};", header, re.S)
    names = ["OP"] + re.findall(r"BCIR_DP_(\w+)", enum.group(1))
    assert {101 + i: n.lower() for i, n in enumerate(names)} == {
        k: v for k, v in REASONS.items() if k > 100
    }, names
    runtime = (_C / "bcir_runtime.h").read_text(encoding="utf-8")
    status = dict((int(v), k.lower()) for k, v in re.findall(r"BCIR_ERR_(\w+) = (\d+)", runtime))
    assert status == {k: v for k, v in REASONS.items() if k < 100}, status


def test_the_pack_fuzz_seeds_start_inside_the_grammar():
    """Every seed of the pack fuzz harness (runtime/c/fuzz_decoder_pack.c) lowers on the harness's
    state as its name says: the planner's packs lower and run every step at the shape their shape
    byte names (one step, and two steps of two micro-batches), the placed ones (PX-4) on the arenas
    their programs reach -- the one a float short refused there with nothing run -- and the near
    misses are refused at their reason: so a campaign starts inside the lowering, the runner, and
    the refusals."""
    if _CC is None:
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.pack_library(Path(tmp) / "pack")
        if path is None:
            return
        seeds = nf.fuzz_pack_seeds()
        assert len(seeds) >= 20 and any(not ok for _, _, ok in seeds)
        names = {bits: name for name, bits in nat._CONTRACTS.items()}
        shapes = set()
        for name, data, ok in seeds:
            config, steps, accumulation, pack, plan = nf.fuzz_pack_parts(data)
            spec, batch, time_ = nf.FUZZ_STATE
            contract = nf.FUZZ_CONTRACTS[config & 7]
            trainer = nat.NativeDecoder(path, spec, batch_size=batch, context_length=time_,
                                        contract=names[contract],
                                        recompute=3 if config & 8 else 0,
                                        workspace_bytes=nf.FUZZ_WORKSPACE if contract & 16 else None)  # fmt: skip
            try:
                rc, why, done, _, _ = _lower(trainer, pack, plan)
                assert (rc == 0) == ok, (name, rc, why, REASONS.get(why))
                if ok:
                    # at the shape its shape byte names, as the harness runs it: every step runs
                    n = batch * time_ * steps * accumulation
                    tokens = [(7 * i + 3) % spec.vocab_size for i in range(n)]
                    targets = [(5 * i + 1) % spec.vocab_size for i in range(n)]
                    words = trainer.lower_pack(pack, plan)
                    lent = nf.fuzz_lent(words, config, trainer.state.plan)
                    if lent is not None:  # PX-4: on the arenas its program reaches
                        trainer.close()
                        trainer = nat.NativeDecoder(path, spec, batch_size=batch,
                                                    context_length=time_, contract=names[contract],
                                                    recompute=3 if config & 8 else 0,
                                                    workspace_bytes=nf.FUZZ_WORKSPACE
                                                    if contract & 16 else None, transient=lent)  # fmt: skip
                    try:
                        events = trainer.run_program(words, tokens, targets, steps=steps,
                                                     gradient_accumulation=accumulation)  # fmt: skip
                    except nat.NativeRunStopped as stopped:
                        assert config & nf.FUZZ_SHORT, (name, stopped)
                        assert (stopped.status, stopped.completed) == (1, 0), (name, stopped)
                        continue
                    assert not config & nf.FUZZ_SHORT, name
                    assert len(events) == steps, (name, events)
                    shapes.add((steps, accumulation))
            finally:
                trainer.close()
        assert shapes == {(1, 1), (2, 2)}, shapes


def test_the_pack_comparison_sees_each_divergence():
    """`pack_problems` -- what `train.plan.pack.mismatch` counts and the corpus witness asserts
    empty -- names each way a pack run can differ from the C step on its case, one problem each (a
    refused lowering, other words than the program's, a stopped run, another state, other
    events), and none in a run that matches: the witness can see what it exists to count."""
    from types import SimpleNamespace

    step = nf.TraceRun([], [], 0, "state", ({"step": 1},))
    ok = nf.PackRun(SimpleNamespace(words=(1, 2)), (1, 2), "state", ({"step": 1},), 0)
    assert nf.pack_problems(step, ok) == []
    for run, expected in (
        (ok._replace(words=(), status=-1, why=105), "refused"),
        (ok._replace(words=(1, 3)), "words"),
        (ok._replace(status=3), "status 3"),
        (ok._replace(state="other"), "state"),
        (ok._replace(events=({"step": 2},)), "events"),
    ):
        found = nf.pack_problems(step, run)
        assert len(found) == 1 and expected in found[0], (run, found)


def test_one_walk_checks_the_pack_and_hands_over_its_lifetimes():
    """`bcir_ep_check_pack_lifetimes` -- the lowering's one walk of the plan's laws, two of them
    quadratic in its records -- returns `bcir_ep_check_pack`'s status on the intact pair and on
    every pack and plan the refusals name, and on an admitted pair hands over exactly the
    lifetimes `bcir_ep_for_each_lifetime` walks, in their order: the check and the walk in one
    pass, no verdict moved."""
    case = nf.trace_corpus()[0]  # v1, the small spec, one step
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.pack_library(Path(tmp) / "pack")
        if path is None:
            return
        trainer, decisions = nf._case_trainer(path, case)
        try:
            tp = training_pack(case.spec, case.batch, case.time, **decisions)
            pack_bytes, plan_bytes, pack, plan = pack_artifacts(tp)
            lib = trainer.lib
            hand = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p)

            def collector(into: list):
                def on(view, _ctx):  # the view's first member is the lifetime's RID (u32)
                    into.append(ct.cast(view, ct.POINTER(ct.c_uint32))[0])
                    return 0

                return hand(on)

            check = lib.bcir_ep_check_pack
            check.argtypes = [ct.c_char_p, ct.c_size_t, ct.c_char_p, ct.c_size_t]
            check.restype = ct.c_int
            one = lib.bcir_ep_check_pack_lifetimes
            one.argtypes = [ct.c_char_p, ct.c_size_t, ct.c_char_p, ct.c_size_t, hand, ct.c_void_p]
            one.restype = ct.c_int
            walk = lib.bcir_ep_for_each_lifetime
            walk.argtypes = [ct.c_char_p, ct.c_size_t, hand, ct.c_void_p]
            walk.restype = ct.c_int
            # each pair with the verifiers' status: the reason when it is theirs (bcir_status,
            # 1-25), else 0 -- a near miss only the lowering's own rules refuse
            pairs = [("intact", pack_bytes, plan_bytes, 0)] + [
                (name, pk, pl, reason if reason < 100 else 0)
                for name, pk, pl, reason in _refusals(tp, pack, plan)
            ]
            admitted = refused = 0
            for name, pk, pl, verdict in pairs:
                handed: list[int] = []
                status = one(pl, len(pl), pk, len(pk), collector(handed), None)
                assert status == check(pl, len(pl), pk, len(pk)) == verdict, (name, status)
                if status:
                    refused += 1
                    continue
                walked: list[int] = []
                assert walk(pl, len(pl), collector(walked), None) == 0, name
                assert handed == walked and handed, name
                if name == "intact":
                    assert handed == [x.rid for x in plan.lifetimes], name
                admitted += 1
            assert admitted + refused == len(pairs) and refused >= 5, (admitted, refused)
        finally:
            trainer.close()


def test_the_lowering_is_lent_an_array_it_can_run():
    """`lower_pack` returns the array('Q') `run_program` runs as it is, or raises the C's refusal
    (`PackRefused`: its status and reason), and `run_pack` is the two: the same state as the
    program's own run, from one lowering or two."""
    case = next(c for c in nf.trace_corpus() if c.steps == 2 and not c.recompute)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.pack_library(Path(tmp) / "pack")
        if path is None:
            return
        trainer, decisions = nf._case_trainer(path, case)
        try:
            tp = training_pack(case.spec, case.batch, case.time, **decisions)
            pack, plan, _, plan_obj = pack_artifacts(tp)
            words = trainer.lower_pack(pack, plan)
            assert isinstance(words, array) and words.typecode == "Q"
            assert list(words) == list(tp.words)
            # a refusal is the C's, status and reason, and nothing to run
            schedule = copy.deepcopy(plan_obj)
            schedule.liveness = "schedule"
            try:
                trainer.lower_pack(pack, encode_plan(schedule))
            except nat.PackRefused as refused:
                assert (refused.status, REASONS.get(refused.why)) == (1, "liveness"), refused
                assert "liveness" in str(refused), refused
            else:
                raise AssertionError("the binding returned a program the C refused")
            tokens, targets, optimizer = nf._case_inputs(case)
            events = trainer.run_pack(pack, plan, tokens, targets, steps=case.steps,
                                      optimizer=optimizer,
                                      gradient_accumulation=case.accumulation)  # fmt: skip
            state = nf.state_digest(trainer)
        finally:
            trainer.close()
        step = nf.trace_case(path, case, trace=False)
        assert (state, events) == (step.state, step.events)
        try:
            nat.NativeDecoder.lower_pack(type("NoPack", (), {"lib": object()})(), pack, plan)
        except ValueError as refused:
            assert "pack=True" in str(refused)
        else:
            raise AssertionError("a library without the lowering lowered a pack")
