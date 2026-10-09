"""NDT-GEM: the native decoder's training step as a GEM+ program, pure Python.

`bcir.hosted.models.native_program` states the C trainer's step from the spec alone. These
tests hold the program to hand-derived values -- the C plan's closed forms, the counted work of
the PR's own fixture worked by hand, G26's roofline and concurrent-live bound recomputed from
their definitions -- and need no compiler. The native rows (the program against the compiled
plan, the GEMM against its binary32 contract, the step against its floor) are in
`test_native_decoder.py`.
"""

from __future__ import annotations

from fractions import Fraction

from bcir.frontends.models.decode import DecoderSpec
from bcir.hosted.models.native import parameter_layout
from bcir.hosted.models.native_program import ARENAS, TRANSIENT, training_step_program
from bcir.kbcir.io_bounds import pebble_bound
from bcir.tests import native_decoder_fixtures as nf

_FIXTURE = (DecoderSpec(16, 8, 2, 2, 16, activation="silu_gate", n_kv_heads=1), 1, 4)


def _closed_form(spec, batch, time):
    """`bcir_decoder_make_plan`'s three counts, restated from its closed forms."""
    n, d, f, v, L = batch * time, spec.d_model, spec.d_ff, spec.vocab_size, spec.n_layers
    k = spec.kv_heads * (d // spec.n_heads)
    nd, nk, nf, nv, att = n * d, n * k, n * f, n * v, n * time * spec.n_heads
    lp = 2 * d + 2 * d * d + 2 * d * k + 3 * d * f
    params = v * d + L * lp + d + (0 if spec.tied_embeddings else v * d)
    la = 5 * nd + 2 * nk + att + 3 * nf
    return params, (L + 1) * nd + L * la + nd + nv, 5 * nd + 2 * nk + 3 * nf + nv


def test_the_arenas_are_the_plans_closed_forms():
    """Every arena of every corpus step sums to the C plan's closed form, and the parameter
    arenas to the binding's HF layout."""
    for spec, batch, time in nf.PLAN_CORPUS:
        got = training_step_program(spec, batch, time).arena_elements()
        params, acts, scratch = _closed_form(spec, batch, time)
        assert got == dict(zip(ARENAS, (params,) * 4 + (acts, scratch))), (spec, batch, time)
        assert params == sum(row[3] for row in parameter_layout(spec))


def test_the_counted_work_is_the_contractions_worked_by_hand():
    """The PR fixture (vocab 16, width 8, 2 heads sharing 1 KV head, 2 layers, ff 16, 1 x 4
    tokens): per layer the seven projections are 4,608 units forward and twice that backward,
    the causal attention 320 forward and 640 backward (80 multiply-adds per contraction), the
    head 1,024 and 2,048 -- 32,640 in all."""
    program = training_step_program(*_FIXTURE)
    assert program.work() == 2 * (4608 + 320) + 1024 + 2 * (2 * 4608 + 640) + 2048 == 32640
    by_kind = {}
    for step in program.steps:
        by_kind[step.name.rsplit(".", 1)[-1]] = by_kind.get(step.name.rsplit(".", 1)[-1], 0) + (
            step.work
        )
    assert by_kind["attention"] == 2 * 320 and by_kind["head"] == 1024
    # every GEMM is one the C issues: the forward linear and the two VJP forms
    assert {(ta, tb) for ta, tb, *_ in program.gemm_shapes()} == {(0, 1), (0, 0), (1, 0)}


def test_every_transient_value_is_written_before_it_is_read():
    """A cache entry or a workspace value is born in the step: its first touch writes it. The
    parameter, moment and gradient arenas (and the tokens) are what the step starts from."""
    program = training_step_program(*_FIXTURE)
    arena = {buf.name: buf.arena for buf in program.buffers}
    seen = set()
    for step in program.steps:
        for vid in step.reads:
            if arena.get(program.values[vid]) in TRANSIENT:
                assert vid in seen, (step.name, program.values[vid])
        seen.update(step.reads)
        seen.update(step.writes)
    live = {program.values[vid] for vid in program.live_out}
    assert live == {b.name for b in program.buffers if b.arena not in TRANSIENT}


def test_the_roofline_is_g26s_over_the_programs_order():
    """Compute only: the counted work over the peak, exactly. With a level: the larger of that
    and the pebble bound at the level's capacity over its bandwidth."""
    program = training_step_program(*_FIXTURE)
    seconds, terms = program.roofline(10**9)
    assert seconds == Fraction(32640, 10**9) and dict(terms) == {"compute": seconds}
    capacity = 64 * 1024
    traffic = pebble_bound(program.order(), program.sizes(), capacity, program.live_out).total
    seconds, terms = program.roofline(10**9, levels=[(capacity, 10**6)])
    assert dict(terms)["level0"] == Fraction(traffic, 10**6)
    assert seconds == max(Fraction(32640, 10**9), Fraction(traffic, 10**6))


def test_the_transient_bound_is_sound_and_the_c_plan_sits_above_it():
    """The concurrent-live bound is at least any one step's live transient bytes, the incumbent
    layout is a layout (so no lower than the bound), and the C plan's two arenas -- every cache
    retained, one span per workspace buffer -- hold at least the incumbent."""
    for spec, batch, time in nf.PLAN_CORPUS:
        program = training_step_program(spec, batch, time)
        items = program.transient_items()
        bound = program.transient_lower_bound()
        for tick in range(len(program.steps)):
            live = sum(i.size for i in items if i.first_tick <= tick < i.last_tick)
            assert live <= bound, (spec, tick)
        incumbent = program.transient_incumbent()
        assert bound <= incumbent.extent <= program.transient_bytes(), (spec, batch, time)
        offsets = incumbent.offsets
        for a in items:  # the incumbent never overlaps two values live at once
            for b in items:
                if a.rid < b.rid and a.conflicts(b):
                    assert (
                        offsets[a.rid] + a.size <= offsets[b.rid]
                        or offsets[b.rid] + b.size <= offsets[a.rid]
                    )


def test_the_program_refuses_what_the_c_plan_refuses():
    for bad in (
        lambda: training_step_program(DecoderSpec(16, 8, 2, 1, 16), 1, 4),  # not SwiGLU
        lambda: training_step_program(_FIXTURE[0], 0, 4),
        lambda: training_step_program(_FIXTURE[0], 1, True),
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("an invalid step was stated")
