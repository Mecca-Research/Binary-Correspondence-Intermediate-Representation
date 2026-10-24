"""PX-1 (TRACE): the native training step's launches held to the ones its step program states.

`bcir.hosted.models.native_launch.run_launches` states, from the step program's buffers and a
trainer's decisions alone, every launch one `bcir_decoder_run_accum` call issues -- each call that
reaches the tensor unit, a provider or the pool, its grouping into jobs, its entry, shapes and
scalars, and where each operand sits. A build defining BCIR_DT_TRACE_PROBE records the launches
the C actually issues. The witness holds the two equal over the trace corpus; the other tests hold
the comparison itself (it names every kind of divergence) and the launch model to the program GEM+
prices (the same GEMMs).
"""

import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

from bcir.frontends.models.decode import DecoderSpec
from bcir.hosted.models.native_launch import (
    ATTN,
    ATTN_VJP,
    GEMM,
    KIND_NAMES,
    PREFLIGHT,
    SPACES,
    Launch,
    launch_mismatches,
    run_launches,
)
from bcir.hosted.models.native_program import training_step_program
from bcir.tests import native_decoder_fixtures as nf

#: The host's C compiler, or None: the trace witness's one capability (a fault sweep `--require`s it,
#: so a host without one is never read as a sweep the witness passed).
_CC = __import__("bcir.toolchain", fromlist=["host_c_compiler"]).host_c_compiler()


def _state_cases(corpus) -> list[int]:
    """One case of the corpus per decision set the state could depend on -- contract, recompute
    or not, workspace tier, threads, provider, a failing bound -- its multi-step one where the
    corpus has it."""
    picked: dict[tuple, int] = {}
    for i, case in enumerate(corpus):
        key = (case.contract, bool(case.recompute), case.workspace, case.threads, case.provider,
               case.preflight)  # fmt: skip
        if key not in picked or case.steps > corpus[picked[key]].steps:
            picked[key] = i
    return sorted(picked.values())


def test_the_c_step_issues_the_planned_launches():
    """Every case of the trace corpus issues exactly the launches its plan states, no record
    dropped -- and the corpus reaches every record kind (the preflight's optional launch
    included), every GEMM entry and both panel operands, so no path is held vacuously (L2). The
    state's arithmetic cannot see what this sees: a group split into its calls, a panel packed
    instead, an operand moved to another dead span all train the same bits. Two properties of the
    instrument are held with it: a trace lent too few words says so, and the trace build trains
    the library's own bits (a record that evaluated an operand twice, or ran its call twice,
    would describe a step the library does not run)."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.trace_library(Path(tmp) / "trace")
        if path is None:
            return  # no C compiler, or no C units (an installed package): a documented skip
        corpus = nf.trace_corpus()
        kinds, entries, panels, problems, states = Counter(), Counter(), Counter(), [], []
        for case in corpus:
            run = nf.trace_case(path, case)
            found = launch_mismatches(run.planned, run.traced)
            if run.dropped or found:
                problems.append((case, run.dropped, found[:2]))
            states.append(run.state)
            for _launch, record in run.traced:
                kinds[record.kind] += 1
                if record.kind == GEMM:
                    entries[record.flags & 15] += 1
                    panels[(record.flags >> 24) & 3] += 1
        assert not problems, problems[:3]
        assert set(kinds) == set(KIND_NAMES), sorted(set(KIND_NAMES) - set(kinds))
        assert kinds[PREFLIGHT] >= 1, "the update whose bound fails never ran its preflight"
        assert set(entries) == set(range(8)), sorted(entries)
        assert panels[1] and panels[2], panels
        # a trace lent too few words holds the first records and counts the rest: a short trace
        # is never read as a short step
        whole = nf.trace_case(path, corpus[0]).traced
        held = len(whole) // 2
        part = nf.trace_case(path, corpus[0], capacity=held)
        assert held and part.traced == whole[:held], (held, len(part.traced))
        assert part.dropped == len(whole) - held, (part.dropped, len(whole) - held)
        # the probe records and changes nothing: the library trains the trace build's bits
        picked = _state_cases(corpus)
        assert len({states[i] for i in picked}) >= len(nf.TRACE_CONTRACTS), "states alike"
        plain = nf.build_library(Path(tmp) / "plain")
        differ = [corpus[i] for i in picked
                  if nf.trace_case(plain, corpus[i], trace=False).state != states[i]]  # fmt: skip
        assert not differ, differ[:3]


def test_a_long_context_lends_the_binary32_attention_the_workspace():
    """ATT-CONTIG: from the copies' context threshold on, a trainer under a binary32 contract lends
    its attention a workspace -- the step's need is the attention's regions -- and both of the
    attention's launches state it, the state's whole workspace; the C step issues exactly the
    launches its plan states, and every program run of the case (the C step's decisions and each
    planner choice) runs to its end with the C step's state and events bit for bit. The trace
    corpus holds the same under the bf16 GEMM contract, whose workspace the binary32 attention is
    lent too; nowhere else in it does a binary32 contract lend one."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.trace_library(Path(tmp) / "trace")
        if path is None:
            return  # no C compiler, or no C units (an installed package): a documented skip
        spec = DecoderSpec(16, 8, 2, 1, 16, activation="silu_gate", n_kv_heads=1)
        case = nf.TraceCase(spec, 1, 1024, "v1")
        problems, runs, lent = [], 0, set()
        for _case, choices, step, run in nf.program_runs(path, [case]):
            runs += 1
            if not choices:
                problems += launch_mismatches(step.planned, step.traced)[:2]
                if step.dropped:
                    problems.append(f"{step.dropped} records dropped")
            problems += [(choices, found) for found in nf.program_problems(step, run)[:2]]
            lent |= {r.refs[8] for launch in run.program for r in launch.records
                     if r.kind in (ATTN, ATTN_VJP)}  # fmt: skip
        assert not problems, problems[:3]
        assert runs == nf.program_run_count([case]) and runs > 1, runs
        assert len(lent) == 1, lent  # one operand, the same on every launch of every program
        space, offset, size = next(iter(lent))
        assert (space, offset) == (SPACES["workspace"], 0) and size > 0, lent


def _as_trace(launches):
    """Planned launches as a trace would hold them: (launch number, record), in order."""
    return [(i + 1, r) for i, launch in enumerate(launches) for r in launch.records]


def test_the_comparison_names_every_divergence():
    """`launch_mismatches` is silent on the plan itself and on an absent optional launch, and names
    each other way a trace can differ: a launch missing or extra, a job split into its calls or two
    merged into one, one operand moved, an optional launch present but different."""
    spec = DecoderSpec(16, 8, 2, 2, 16, activation="silu_gate", n_kv_heads=1)
    planned = run_launches(spec, 1, 4, contract=0)
    required = [launch for launch in planned if not launch.optional]
    trace = _as_trace(required)
    assert launch_mismatches(planned, trace) == []
    # the preflight present, as the plan states it, is no difference either
    with_preflight = _as_trace(planned)
    assert launch_mismatches(planned, with_preflight) == []
    group = next(i for i, launch in enumerate(required) if len(launch.records) == 3)
    single = next(i for i, launch in enumerate(required) if len(launch.records) == 1)
    split = (
        required[:group] + [Launch((r,)) for r in required[group].records] + required[group + 1 :]
    )
    merged = required[:single] + [Launch(required[single].records + required[single + 1].records)]
    merged += required[single + 2 :]
    moved = list(required)
    record = moved[single].records[0]
    space, offset, size = record.refs[0]
    moved[single] = Launch((replace(record, refs=((space, offset + 4, size),) + record.refs[1:]),))
    preflight = next(i for i, launch in enumerate(planned) if launch.optional)
    wrong = list(planned)
    pre = wrong[preflight].records[0]
    wrong[preflight] = Launch((replace(pre, dims=(pre.dims[0] + 1,) + pre.dims[1:]),))
    for name, launches in (
        ("missing", required[:single] + required[single + 1 :]),
        ("extra", required[:single] + [required[single]] + required[single:]),
        ("split", split),
        ("merged", merged),
        ("moved", moved),
        ("optional but different", wrong),
    ):
        assert launch_mismatches(planned, _as_trace(launches)), name


def test_the_launches_issue_the_programs_gemms():
    """The launches issue the step program's GEMMs, shape for shape -- the program GEM+ prices
    (roofline, work) is the one the launches run -- under every recompute set: the backward's
    re-run forward included, the panels' groups and a provider's calls the same products."""
    for spec, batch, time_ in nf.PLAN_CORPUS:
        for recompute in dict.fromkeys((0, (1 << spec.n_layers) - 1, 1)):
            program = training_step_program(spec, batch, time_, recompute)
            want = Counter(g for step in program.steps for g in step.gemms)
            for contract, provider in ((0, False), (16, False), (4, True)):
                launches = run_launches(spec, batch, time_, contract=contract,
                                        recompute=recompute, provider_mm=provider)  # fmt: skip
                got = Counter(
                    r.dims[:5] for launch in launches for r in launch.records if r.kind == GEMM
                )
                assert got == want, (spec, recompute, contract, got - want, want - got)


def test_the_launch_model_refuses_what_it_cannot_state():
    """A call shape that is no call, a decoder the native step is not, and a recompute set naming
    a layer the model lacks are refused, never stated as launches."""
    spec = DecoderSpec(16, 8, 2, 2, 16, activation="silu_gate", n_kv_heads=1)
    for bad in (
        lambda: run_launches(spec, 1, 4, contract=0, steps=0),
        lambda: run_launches(spec, 1, 4, contract=0, accumulation=0),
        lambda: run_launches(spec, 1, 4, contract=0, steps=True),
        lambda: run_launches(DecoderSpec(16, 8, 2, 2, 16), 1, 4, contract=0),
        lambda: run_launches(spec, 1, 4, contract=0, recompute=0b100),
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("the launch model stated launches for a refused call")
    names = {launch.records[0].kind for launch in run_launches(spec, 1, 4, contract=0)}
    assert KIND_NAMES[min(names)] == "scan"
