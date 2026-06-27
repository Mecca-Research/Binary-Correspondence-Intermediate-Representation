"""A1 -- per-group Q8<->float32<->Q8 quantization bridge + its R17 accuracy contract.

Covers: round-trip error is always within each group's static bound (the property R17 certifies);
per-GROUP scaling beats per-TENSOR on heterogeneous data (the SOTA reason to do it); the codes always
fit their `_BitInt(N)` lane (|code| <= code_max -- no overflow); determinism; edge/adversarial inputs
(all-zero, single outlier, extreme dynamic range, non-finite rejection); and the R17 law over quantized
claims (a quantize bridge is 1 ULP; a quantized reduction is quant + reduce, so a tight tolerance forces
both compensation and budgeting the quant step). All pure-Python + deterministic (no toolchain needed)."""

import random

from bcir.kbcir.precision import accuracy_bound, quantization_error_bound, reduction_error_bound
from bcir.kbcir.quantize import (QGroup, code_max, dequantize, max_abs_error, quantize_group,
                                 quantize_per_group, roundtrip)
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.verify import verify_accuracy

_EPS = 1e-9


# --- correctness: the round-trip stays within each group's static bound ----------

def test_roundtrip_is_within_the_group_error_bound():
    vals = [0.0, 0.5, -0.5, 1.0, -1.0, 0.123, -0.987, 0.4999, 12.5, -7.25]
    g = quantize_group(vals, bits=8)
    deq = g.dequantize()
    bound = g.error_bound(rounding="nearest")
    assert all(abs(v - d) <= bound + _EPS for v, d in zip(vals, deq))
    # the max-magnitude element sets the scale and must NOT saturate (only the scale, never a clip).
    assert max(abs(c) for c in g.codes) <= code_max(8)


def test_truncate_bound_is_one_full_step():
    vals = [0.9, -0.9, 0.49, -0.49, 3.3, -2.2]
    g = quantize_group(vals, bits=6, rounding="truncate")
    step = g.error_bound(rounding="truncate")
    assert all(abs(v - d) <= step + _EPS for v, d in zip(vals, g.dequantize()))


def test_dequantize_length_matches_and_is_a_shift():
    vals = [float(i) - 8 for i in range(17)]                 # 17 elements, group_size 8 -> 3 blocks (8,8,1)
    groups = quantize_per_group(vals, group_size=8, bits=8)
    assert len(groups) == 3 and tuple(len(g.codes) for g in groups) == (8, 8, 1)
    assert len(dequantize(groups)) == len(vals)
    # dequant is code * 2**scale_exp -- exact (no rounding) for in-range codes.
    for g in groups:
        for c, d in zip(g.codes, g.dequantize()):
            assert d == c * g.scale


# --- the SOTA reason for per-group scaling: it beats per-tensor on heterogeneous data ---

def test_per_group_beats_per_tensor_on_a_wide_dynamic_range():
    G = 16
    big = [1000.0 + (i % 7) for i in range(G)]               # ~1e3 magnitudes
    tiny = [0.01 * (1 + (i % 5)) for i in range(G)]          # ~1e-2 magnitudes
    vals = big + tiny

    def small_block_err(group_size):
        deq = roundtrip(vals, group_size, bits=4)
        return max(abs(v - d) for v, d in zip(vals[G:], deq[G:]))   # error on the TINY block

    per_tensor = small_block_err(len(vals))                 # one global scale (set by the big block)
    per_group = small_block_err(G)                          # the tiny block gets its own fine scale
    # the tiny block, swamped by the big block's coarse global scale, is far more accurate per-group.
    assert per_group < per_tensor
    assert per_group * 10 < per_tensor                      # and by a wide margin, not a coin-flip


def test_finer_groups_never_increase_error():
    rng = random.Random(20260627)
    vals = [rng.uniform(-50, 50) * (10 ** rng.randint(-3, 3)) for _ in range(64)]
    whole = max_abs_error(vals, group_size=64, bits=6)      # per-tensor
    g16 = max_abs_error(vals, group_size=16, bits=6)
    g4 = max_abs_error(vals, group_size=4, bits=6)
    assert g4 <= g16 + _EPS <= whole + _EPS                 # monotone: finer grouping cannot hurt


# --- edge / adversarial inputs (red-team) ----------------------------------------

def test_all_zero_group_is_exact():
    g = quantize_group([0.0, 0.0, 0.0], bits=8)
    assert g.codes == (0, 0, 0) and g.dequantize() == [0.0, 0.0, 0.0]


def test_single_outlier_sets_the_scale_and_nothing_saturates():
    vals = [1e6] + [0.0] * 31                               # one huge value dominates the block
    g = quantize_group(vals, bits=8)
    assert max(abs(c) for c in g.codes) <= code_max(8)      # the outlier itself does not clip
    assert abs(vals[0] - g.dequantize()[0]) <= g.error_bound() + _EPS


def test_extreme_dynamic_range_does_not_crash_or_overflow():
    for v in (1e30, 1e-30, -1e30, 3.4e38, 1e-38):
        g = quantize_group([v, v / 2, -v], bits=8)
        assert max(abs(c) for c in g.codes) <= code_max(8)
        assert all(abs(x) != float("inf") for x in g.dequantize())


def test_non_finite_inputs_are_rejected_at_the_bridge_boundary():
    for bad in (float("inf"), float("-inf"), float("nan")):
        try:
            quantize_group([1.0, bad, 2.0], bits=8)
            assert False, f"expected ValueError on {bad}"
        except ValueError:
            pass


def test_degenerate_arguments_raise():
    for bits in (-1, 0, 1):
        try:
            code_max(bits); assert False, f"bits={bits} should raise"
        except ValueError:
            pass
    try:
        quantize_per_group([1.0], group_size=0, bits=8); assert False, "group_size 0 should raise"
    except ValueError:
        pass
    try:
        quantize_group([1.0], bits=8, rounding="bogus"); assert False, "bad rounding should raise"
    except ValueError:
        pass


def test_empty_input_is_empty_output():
    assert quantize_per_group([], group_size=8, bits=8) == []
    assert roundtrip([], 8, 8) == []


# --- determinism ------------------------------------------------------------------

def test_quantization_is_deterministic():
    rng = random.Random(7)
    vals = [rng.uniform(-100, 100) for _ in range(50)]
    a = quantize_per_group(vals, group_size=8, bits=8)
    b = quantize_per_group(vals, group_size=8, bits=8)
    assert a == b                                           # frozen dataclasses compare by value
    assert all(isinstance(c, int) for g in a for c in g.codes)


# --- stress / fuzz: invariants hold over a broad random campaign -----------------

def test_fuzz_roundtrip_bound_and_lane_fit_hold_everywhere():
    rng = random.Random(0xA1)
    for _ in range(4000):
        n = rng.randint(1, 40)
        bits = rng.choice([2, 3, 4, 5, 8, 12, 16])
        gs = rng.randint(1, n)
        rounding = rng.choice(["nearest", "truncate"])
        scale = 10.0 ** rng.randint(-6, 6)
        vals = [rng.uniform(-1.0, 1.0) * scale for _ in range(n)]
        groups = quantize_per_group(vals, gs, bits, rounding=rounding)
        cmax = code_max(bits)
        deq = dequantize(groups)
        assert len(deq) == n
        # (1) every code fits its _BitInt(bits) lane -- the A1 packing invariant.
        assert all(-cmax <= c <= cmax for g in groups for c in g.codes)
        # (2) every element's round-trip error is within its group's static bound (the R17 property).
        i = 0
        for g in groups:
            b = g.error_bound(rounding=rounding)
            for v, d in zip(vals[i:i + gs], g.dequantize()):
                assert abs(v - d) <= b + max(_EPS, b * 1e-9)
            i += gs


# --- R17: the accuracy contract over quantized claims ----------------------------

def _module(claim, count):
    m = Module(name="q")
    m.add_resource(Resource(rid=50, domain=Domain.RAM, shape=(count,)))
    m.add_resource(Resource(rid=51, domain=Domain.RAM, shape=(1,)))
    m.add_phase(Phase(phase_id=0, deps=(), claims=[claim]))
    return m


def _claim(op, *, count=1000, precision="", tol=0, qbits=0):
    return Claim(id=6000, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT, count=count,
                 rd=(50,), wr=(51,), op=op, domain=Domain.RAM, precision=precision, tolerance_ulp=tol,
                 quantized_bits=qbits)


def test_quantization_error_bound_is_one_ulp():
    assert quantization_error_bound() == 1
    assert quantization_error_bound(rounding="truncate") == 1


def test_accuracy_bound_composes_quant_with_the_op_error():
    # a bare quantize bridge claim: 1 ULP at its grid.
    assert accuracy_bound(_claim("quantize.q8")) == 1
    # a dense 1000-term reduction is unchanged (non-disturbance): count naive, 1 compensated.
    assert accuracy_bound(_claim("reduce.add", count=1000)) == 1000
    assert accuracy_bound(_claim("reduce.add", count=1000), compensated=True) == 1
    # a QUANTIZED 1000-term reduction adds the bridge step on top of the reduction error.
    assert accuracy_bound(_claim("reduce.add", count=1000, qbits=4)) == 1000 + 1
    assert accuracy_bound(_claim("reduce.add", count=1000, qbits=4), compensated=True) == 1 + 1
    # a quantized elementwise op: 1 (op) + 1 (bridge).
    assert accuracy_bound(_claim("vector.add", count=8, qbits=8)) == 2


def test_r17_on_a_quantized_reduction_forces_compensation_and_budgets_the_quant_step():
    # tol=2 on a 1000-term quantized reduction: naive bound 1001 > 2 -> fires.
    diags = verify_accuracy(_module(_claim("reduce.add", count=1000, tol=2, qbits=8), 1000))
    assert [d.law for d in diags] == ["R17"]
    # compensated: bound 1 (reduce) + 1 (quant) = 2 <= 2 -> legal.
    assert verify_accuracy(_module(_claim("reduce.add", count=1000, precision="compensated", tol=2, qbits=8), 1000)) == []
    # tol=1 is IMPOSSIBLE for a quantized reduction: even compensated, the quant step alone costs 1 ULP
    # on top of the reduction's 1 -> bound 2 > 1 -> R17 fires (an honest "you cannot have this").
    diags = verify_accuracy(_module(_claim("reduce.add", count=1000, precision="compensated", tol=1, qbits=8), 1000))
    assert [d.law for d in diags] == ["R17"]


def test_r17_is_a_noop_for_dense_claims_unchanged():
    # the non-disturbance guard: a dense (quantized_bits=0) claim with no tolerance is untouched.
    assert verify_accuracy(_module(_claim("reduce.add", count=1000), 1000)) == []
    assert accuracy_bound(_claim("reduce.add", count=1000)) == reduction_error_bound(1000)
