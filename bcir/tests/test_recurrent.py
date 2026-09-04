"""E4 (ML-breadth) -- RECURRENT CELLS (RNN / LSTM / GRU): the TWO-TIER correctness gate, the FOURTH ML-breadth
slice after E1 (OLS), E2 (PCA), E3 (the Transformer block). The design mirrors M1's two-path losses, forced by
the B3 autodiff's CLOSED primitive set (no transcendentals):

  * TIER A -- the closed-set relu-RNN, trainable end-to-end via the EXISTING ``unroll_scan`` + ``grad`` (= literal
    BPTT). THE HEADLINE GATE: build the unrolled DAG over a short sequence and assert the BPTT gradient (w.r.t.
    the inputs AND the weights) matches ``finite_difference_grad`` via ``gradients_match`` -- a real trainable
    recurrent net whose BPTT is the existing reverse-mode AD, verified by finite differences. Inputs are kept
    AWAY from the relu kink (z != 0) so the a.e. select-derivative equals the finite difference. A tiny
    end-to-end training run (reusing training.py) drives a loss down.
  * TIER B -- LSTM and GRU cells (transcendental tanh/sigmoid gates), the M1 closed-form-seed treatment: a numeric
    forward reference (checked against a hand-computed small example) + closed-form analytic gradients (checked
    against CENTRAL finite differences, numeric since tanh/sigmoid are not closed Tape primitives).

Also covers: the gate-range property (every sigmoid in (0,1), every tanh in (-1,1)); a temporal-dependence check
(h_T genuinely depends on x_0 -- memory across time, dh_T/dx_0 != 0); the Q8 bridge tracks the reference within
the R17 input bound AND equals the cell of the round-tripped input (clean bridge); the LSTM C kernel compiles +
reproduces the reference + routes tanhf/expf through the libm edge; the tanhf/expf -> -lm link rule is unchanged
(NO linkflags change); the shape/dtype check accepts valid claims + rejects malformed ones (NOT a new global
R-law); and the module imports no verifier / emits no Diagnostic (the two-truth quarantine, AST-inspected). All
deterministic + pure-Python on the oracle rail.
"""

import ast
import math
import os
import random
import shutil
import subprocess
import tempfile

from bcir.frontends.cfront.linkflags import NO_FLAG, library_for_callee
from bcir.toolchain import host_link_args
from bcir.kbcir.autodiff import (
    Tape,
    evaluate,
    finite_difference_grad,
    grad,
    gradients_match,
    max_grad_error,
    unroll_scan,
)
from bcir.kbcir.precision import quantization_error_bound
from bcir.kbcir.recurrent import (
    GruParams,
    LstmParams,
    RnnParams,
    central_difference_jacobian,
    check_recurrent,
    gate_range_residual,
    gru_cell_grads,
    gru_cell_reference,
    gru_states,
    gru_unroll,
    gru_via_bridge,
    lstm_cell_grads,
    lstm_cell_reference,
    lstm_states,
    lstm_unroll,
    lstm_via_bridge,
    max_jacobian_error,
    rnn_relu_reference,
    rnn_relu_states,
    rnn_relu_step,
    rnn_relu_unroll,
    rnn_scalar_step,
    sigmoid,
    sigmoid_prime_from_value,
    tanh_prime_from_value,
    temporal_dependence,
)
from bcir.lower.c_kernel import emit_lstm_cell_c
from bcir.tests._convergence import assert_converged


def _vec(rng, n, lo=-0.5, hi=0.5):
    return [rng.uniform(lo, hi) for _ in range(n)]


# ============================================================================================================
# 1. TIER A -- the closed-set relu-RNN: BPTT (unroll + grad) == finite differences (THE HEADLINE).
# ============================================================================================================


def _rnn_params(rng, di, dh, *, positive_bias=True):
    b = _vec(rng, dh, 0.1, 0.5) if positive_bias else _vec(rng, dh)
    return RnnParams(_vec(rng, dh * di), _vec(rng, dh * dh), b, di, dh)


def _build_rnn_dag(tape, x_names, h0_val, params):
    """Build the unrolled relu-RNN DAG and a scalar readout (sum of the final hidden state). Returns
    (readout_node, final_hidden_ids)."""
    h0_ids = tuple(tape.const(h0_val) for _ in range(params.hidden_dim))
    hT = rnn_relu_unroll(tape, x_names, h0_ids, params)
    readout = hT[0]
    for nid in hT[1:]:
        readout = tape.add(readout, nid)
    return readout, hT


def test_tier_a_evaluate_matches_the_plain_float_reference():
    # The Tape-built unrolled RNN, evaluated, equals the independent plain-float reference forward.
    rng = random.Random(0xE401)
    di, dh, T = 2, 3, 4
    params = _rnn_params(rng, di, dh)
    x_names = [[f"x{t}_{j}" for j in range(di)] for t in range(T)]
    env = {f"x{t}_{j}": rng.uniform(0.3, 0.9) for t in range(T) for j in range(di)}
    tape = Tape()
    _, hT = _build_rnn_dag(tape, x_names, 0.2, params)
    x_seq = [env[f"x{t}_{j}"] for t in range(T) for j in range(di)]
    ref = rnn_relu_reference(x_seq, [0.2] * dh, params)
    got = [evaluate(tape, nid, env) for nid in hT]
    assert all(abs(g - r) < 1e-12 for g, r in zip(got, ref)), (got, ref)


def test_tier_a_bptt_matches_finite_difference_wrt_inputs():
    # THE HEADLINE: grad (= reverse-mode AD over the unrolled DAG = BPTT) of a scalar readout w.r.t. EVERY input
    # equals the central finite difference. Inputs kept away from the relu kink (positive bias + positive inputs
    # -> z != 0), so the a.e. select-derivative matches FD.
    rng = random.Random(0xE402)
    di, dh, T = 2, 3, 5
    params = _rnn_params(rng, di, dh)
    x_names = [[f"x{t}_{j}" for j in range(di)] for t in range(T)]
    env = {f"x{t}_{j}": rng.uniform(0.4, 1.0) for t in range(T) for j in range(di)}
    tape = Tape()
    readout, _ = _build_rnn_dag(tape, x_names, 0.3, params)
    g = grad(tape, readout, env)
    fd = finite_difference_grad(tape, readout, env)
    assert gradients_match(g.grads, fd), (
        f"BPTT != FD: max_abs_err={max_grad_error(g.grads, fd)} analytic={g.grads} fd={fd}"
    )


def test_tier_a_bptt_matches_finite_difference_wrt_weights():
    # BPTT also differentiates the WEIGHTS: rebuild the step using var nodes for the weights, then grad w.r.t.
    # them == FD. This is the "trainable" half -- the gradient that an optimizer would consume.
    rng = random.Random(0xE403)
    di, dh, T = 2, 2, 4
    # weights as named vars so they are differentiation targets.
    wx_names = [f"wx{o}_{j}" for o in range(dh) for j in range(di)]
    wh_names = [f"wh{o}_{k}" for o in range(dh) for k in range(dh)]
    b_names = [f"b{o}" for o in range(dh)]
    wx_val = _vec(rng, dh * di)
    wh_val = _vec(rng, dh * dh)
    b_val = _vec(rng, dh, 0.1, 0.5)
    x_vals = [[rng.uniform(0.4, 1.0) for _ in range(di)] for _ in range(T)]

    def build(tape):
        # bind the weight + bias vars, build the unrolled recurrence by hand using the var-backed weights.
        wx = [tape.var(n) for n in wx_names]
        wh = [tape.var(n) for n in wh_names]
        bb = [tape.var(n) for n in b_names]
        zero = tape.const(0.0)
        h = [tape.const(0.2) for _ in range(dh)]
        for t in range(T):
            xc = [tape.const(v) for v in x_vals[t]]
            new_h = []
            for o in range(dh):
                z = tape.const(0.0)
                for j in range(di):
                    z = tape.add(z, tape.mul(wx[o * di + j], xc[j]))
                for k in range(dh):
                    z = tape.add(z, tape.mul(wh[o * dh + k], h[k]))
                z = tape.add(z, bb[o])
                new_h.append(tape.select(z, z, zero))
            h = new_h
        out = h[0]
        for nid in h[1:]:
            out = tape.add(out, nid)
        return out

    tape = Tape()
    out = build(tape)
    env = dict(zip(wx_names, wx_val))
    env.update(zip(wh_names, wh_val))
    env.update(zip(b_names, b_val))
    g = grad(tape, out, env)
    fd = finite_difference_grad(tape, out, env)
    assert gradients_match(g.grads, fd), (
        f"weight-BPTT != FD: max_abs_err={max_grad_error(g.grads, fd)}"
    )


def test_tier_a_scalar_recurrence_through_unroll_scan_verbatim():
    # The single-hidden-unit recurrence folded by autodiff.unroll_scan VERBATIM (the EXISTING scan API), proving
    # the explicit vector unroll is the same finite composition the scalar scan builds; grad == BPTT == FD.
    rng = random.Random(0xE404)
    T = 5
    tape = Tape()
    xs = tuple(tape.var(f"s{t}") for t in range(T))
    init = tape.const(0.1)
    step = lambda tp, c, x: rnn_scalar_step(tp, c, x, w_h=0.5, w_x=0.7, b=0.2)  # noqa: E731
    out = unroll_scan(tape, step, init, xs)
    env = {f"s{t}": rng.uniform(0.3, 1.0) for t in range(T)}
    g = grad(tape, out, env)
    fd = finite_difference_grad(tape, out, env)
    assert gradients_match(g.grads, fd), max_grad_error(g.grads, fd)


def test_tier_a_step_builder_validates_and_is_closed_set():
    # rnn_relu_step builds ONLY closed-set ops (const/var/dot/add/select), so the resulting tape contains no
    # transcendental node op. Also validates input lengths.
    rng = random.Random(0xE405)
    di, dh = 3, 2
    params = _rnn_params(rng, di, dh)
    tape = Tape()
    h_prev = tuple(tape.var(f"h{i}") for i in range(dh))
    x_ids = tuple(tape.var(f"x{j}") for j in range(di))
    new_h = rnn_relu_step(tape, h_prev, x_ids, params)
    assert len(new_h) == dh
    ops = {tape.node(nid).op for nid in range(tape.node_count())}
    assert ops <= {"const", "var", "dot", "add", "select", "mul", "sub", "neg", "div"}, ops
    # bad lengths raise.
    for bad in [(h_prev[:1], x_ids), (h_prev, x_ids[:1])]:
        try:
            rnn_relu_step(tape, bad[0], bad[1], params)
            assert False, "should raise"
        except ValueError:
            pass


def test_tier_a_rnn_params_validate():
    for args in [
        ([0.0] * 2, [0.0] * 4, [0.0] * 2, 0, 2),  # input_dim < 1
        ([0.0] * 2, [0.0] * 4, [0.0] * 2, 1, 0),  # hidden_dim < 1
        ([0.0] * 3, [0.0] * 4, [0.0] * 2, 1, 2),  # bad w_x length
        ([0.0] * 2, [0.0] * 3, [0.0] * 2, 1, 2),  # bad w_h length
        ([0.0] * 2, [0.0] * 4, [0.0] * 1, 1, 2),
    ]:  # bad b length
        try:
            RnnParams(*args)
            assert False, f"{args} should raise"
        except ValueError:
            pass


def test_tier_a_reference_is_side_effect_free():
    rng = random.Random(0xE406)
    di, dh, T = 2, 2, 3
    params = _rnn_params(rng, di, dh)
    x = _vec(rng, T * di, 0.2, 1.0)
    x0, h0 = list(x), [0.1, 0.2]
    h0c = list(h0)
    rnn_relu_reference(x, h0, params)
    assert x == x0 and h0 == h0c


def test_tier_a_end_to_end_training_drives_loss_down():
    # A tiny end-to-end run: train a recurrent readout to regress a scalar target via the M3 training loop (reuse,
    # don't reinvent). We expose the RNN readout as a training "model" over a Tape; the loss must DECREASE. This
    # demonstrates the closed-set RNN is genuinely trainable through the existing unroll+grad+optimizer stack.
    from bcir.kbcir.training import Dataset, train

    rng = random.Random(0xE407)
    di, dh, T = 1, 2, 3
    # fixed RNN weights baked as constants; the trainable params are a linear readout over the final state.
    wx = _vec(rng, dh * di, 0.3, 0.7)
    wh = _vec(rng, dh * dh, 0.1, 0.3)
    b = _vec(rng, dh, 0.1, 0.3)
    rnn = RnnParams(wx, wh, b, di, dh)

    # dataset: each example is a length-T sequence (flattened); label is a smooth function of the final state.
    seqs, ys = [], []
    for _ in range(24):
        s = [rng.uniform(0.4, 1.0) for _ in range(T * di)]
        hT = rnn_relu_reference(s, [0.0] * dh, rnn)
        seqs.append(tuple(s))
        ys.append(0.7 * hT[0] - 0.3 * hT[1] + 0.05)  # a linear target over the final state
    ds = Dataset(tuple(seqs), tuple(ys))

    # model: build the (constant-weight) RNN forward per example, then a trainable linear readout r0*hT0+r1*hT1+r2.
    def model(tape, param_names, X):
        out = []
        r = [tape.var(nm) for nm in param_names]  # r0, r1, bias
        for row in X:
            x_names = []  # encode the sequence as per-step const nodes
            h = [tape.const(0.0) for _ in range(dh)]
            for t in range(T):
                xc = [tape.const(row[t * di + j]) for j in range(di)]
                new_h = []
                zero = tape.const(0.0)
                for o in range(dh):
                    z = tape.const(b[o])
                    for j in range(di):
                        z = tape.add(z, tape.mul(tape.const(wx[o * di + j]), xc[j]))
                    for k in range(dh):
                        z = tape.add(z, tape.mul(tape.const(wh[o * dh + k]), h[k]))
                    new_h.append(tape.select(z, z, zero))
                h = new_h
            pred = tape.add(tape.add(tape.mul(r[0], h[0]), tape.mul(r[1], h[1])), r[2])
            out.append(pred)
        return out

    names = ["r0", "r1", "r2"]
    res = train(
        model,
        [0.0, 0.0, 0.0],
        ds,
        loss="mse",
        optimizer="sgd",
        epochs=40,
        batch_size=8,
        lr=0.3,
        seed=1,
        param_names=names,
    )
    # the shared convergence gate: decreased + absolute threshold + substantial drop.
    assert_converged(res.train_loss, final_below=1e-2, name="E4 Tier-A readout")


# ============================================================================================================
# 2. TIER B -- the gate helpers: sigmoid guard + the closed-form gate derivatives.
# ============================================================================================================


def test_sigmoid_is_guarded_and_in_range():
    assert sigmoid(0.0) == 0.5
    assert abs(sigmoid(1000.0) - 1.0) < 1e-12  # no overflow (saturates to 1.0)
    assert abs(sigmoid(-1000.0) - 0.0) < 1e-12  # no overflow (underflows to 0.0)
    assert 0.0 <= sigmoid(1000.0) <= 1.0 and 0.0 <= sigmoid(-1000.0) <= 1.0  # always within [0, 1]
    rng = random.Random(0xE408)
    for _ in range(200):
        x = rng.uniform(-15, 15)  # a realistic pre-activation range (no float saturation)
        s = sigmoid(x)
        assert 0.0 < s < 1.0  # strictly in (0, 1) away from float saturation
        # the value matches the naive form (the guarded form is algebraically exact for |x| in range).
        assert abs(s - 1.0 / (1.0 + math.exp(-x))) < 1e-12


def test_gate_derivatives_match_finite_difference():
    # sigma'(x) = s(1-s) and tanh'(x) = 1 - t^2 as closed forms vs the numeric central difference of the gate.
    rng = random.Random(0xE409)
    for _ in range(100):
        x = rng.uniform(-4, 4)
        eps = 1e-6
        ds = (sigmoid(x + eps) - sigmoid(x - eps)) / (2 * eps)
        assert abs(sigmoid_prime_from_value(sigmoid(x)) - ds) < 1e-7, x
        dt = (math.tanh(x + eps) - math.tanh(x - eps)) / (2 * eps)
        assert abs(tanh_prime_from_value(math.tanh(x)) - dt) < 1e-7, x


# ============================================================================================================
# 3. TIER B.1 -- the LSTM cell: hand-computed forward + analytic grads vs central FD + the unroll.
# ============================================================================================================


def _lstm_params(rng, di, dh):
    def w(n):
        return _vec(rng, n)

    return LstmParams(
        w(dh * di),
        w(dh * dh),
        w(dh),
        w(dh * di),
        w(dh * dh),
        w(dh),
        w(dh * di),
        w(dh * dh),
        w(dh),
        w(dh * di),
        w(dh * dh),
        w(dh),
        di,
        dh,
    )


def test_lstm_forward_matches_a_hand_computed_example():
    # A 1x1 LSTM with W_*=1, U_*=0, b_*=0, x=[0.5], h_prev=[0], c_prev=[0]: every gate pre-activation is 0.5.
    p = LstmParams(
        [1.0], [0.0], [0.0], [1.0], [0.0], [0.0], [1.0], [0.0], [0.0], [1.0], [0.0], [0.0], 1, 1
    )
    h, c = lstm_cell_reference([0.5], [0.0], [0.0], p)
    f = i = o = sigmoid(0.5)
    g = math.tanh(0.5)
    c_exp = f * 0.0 + i * g
    h_exp = o * math.tanh(c_exp)
    assert abs(c[0] - c_exp) < 1e-15 and abs(h[0] - h_exp) < 1e-15, (h, c, h_exp, c_exp)


def test_lstm_analytic_gradients_match_central_finite_difference():
    # The closed-form analytic Jacobian blocks vs CENTRAL finite differences (numeric, since tanh/sigmoid are not
    # closed Tape primitives). All six blocks (d{h,c}/d{x,h_prev,c_prev}) to a tight tolerance.
    rng = random.Random(0xE40A)
    di, dh = 3, 2
    p = _lstm_params(rng, di, dh)
    x, h_prev, c_prev = _vec(rng, di), _vec(rng, dh), _vec(rng, dh)
    G = lstm_cell_grads(x, h_prev, c_prev, p)

    def h_of(xx, hh, cc):
        return lstm_cell_reference(xx, hh, cc, p)[0]

    def c_of(xx, hh, cc):
        return lstm_cell_reference(xx, hh, cc, p)[1]

    checks = [
        ("dh_dx", lambda v: h_of(v, h_prev, c_prev), x, di),
        ("dc_dx", lambda v: c_of(v, h_prev, c_prev), x, di),
        ("dh_dh_prev", lambda v: h_of(x, v, c_prev), h_prev, dh),
        ("dc_dh_prev", lambda v: c_of(x, v, c_prev), h_prev, dh),
        ("dh_dc_prev", lambda v: h_of(x, h_prev, v), c_prev, dh),
        ("dc_dc_prev", lambda v: c_of(x, h_prev, v), c_prev, dh),
    ]
    for name, fwd, base, n_in in checks:
        fd = central_difference_jacobian(fwd, base, dh)
        err = max_jacobian_error(G[name], fd)
        assert err < 1e-6, (name, err)


def test_lstm_gate_ranges_and_side_effect_free():
    # Every sigmoid gate in (0,1), every tanh in (-1,1) -- the defining gate-nonlinearity property (independent
    # sanity rail on the transcendentals). And the reference does not mutate its inputs.
    rng = random.Random(0xE40B)
    di, dh = 4, 3
    p = _lstm_params(rng, di, dh)
    x, h_prev, c_prev = _vec(rng, di, -3, 3), _vec(rng, dh, -3, 3), _vec(rng, dh, -3, 3)
    x0, h0, c0 = list(x), list(h_prev), list(c_prev)
    h, c = lstm_cell_reference(x, h_prev, c_prev, p)
    assert x == x0 and h_prev == h0 and c_prev == c0  # side-effect-free
    # reconstruct the gate values to range-check them (independent recompute of the pre-activations).
    from bcir.kbcir.recurrent import _row_dot

    sigs, tanhs = [], []
    for u in range(dh):
        for W, U, b in ((p.W_f, p.U_f, p.b_f), (p.W_i, p.U_i, p.b_i), (p.W_o, p.U_o, p.b_o)):
            sigs.append(sigmoid(_row_dot(W, x, u, di) + _row_dot(U, h_prev, u, dh) + b[u]))
        a_g = _row_dot(p.W_g, x, u, di) + _row_dot(p.U_g, h_prev, u, dh) + p.b_g[u]
        tanhs.append(math.tanh(a_g))
        tanhs.append(math.tanh(c[u]))
    assert gate_range_residual(sigs, 0.0, 1.0) == 0.0
    assert gate_range_residual(tanhs, -1.0, 1.0) == 0.0


def test_lstm_unroll_is_repeated_cell_and_validates():
    rng = random.Random(0xE40C)
    di, dh, T = 2, 3, 4
    p = _lstm_params(rng, di, dh)
    x_seq = _vec(rng, T * di)
    h0, c0 = _vec(rng, dh), _vec(rng, dh)
    # unroll == manually stepping the cell T times.
    h, c = list(h0), list(c0)
    for t in range(T):
        h, c = lstm_cell_reference(x_seq[t * di : (t + 1) * di], h, c, p)
    hu, cu = lstm_unroll(x_seq, h0, c0, p)
    assert all(abs(a - b) < 1e-15 for a, b in zip(h, hu))
    assert all(abs(a - b) < 1e-15 for a, b in zip(c, cu))
    # states trajectory has T+1 entries, first is the init.
    traj = lstm_states(x_seq, h0, c0, p)
    assert len(traj) == T + 1 and traj[0][0] == [float(v) for v in h0]
    try:
        lstm_unroll(x_seq + [0.0], h0, c0, p)  # not a multiple of input_dim
        assert False
    except ValueError:
        pass


def test_lstm_temporal_dependence_h_T_depends_on_x0():
    # The MEMORY check: dh_T/dx_0 is genuinely NONZERO -> the final state remembers the first input across time.
    rng = random.Random(0xE40D)
    di, dh, T = 2, 3, 5
    p = _lstm_params(rng, di, dh)
    x_seq, h0, c0 = _vec(rng, T * di), _vec(rng, dh), _vec(rng, dh)

    def fwd_x0(x0blk):
        xs = list(x_seq)
        xs[0:di] = x0blk
        return lstm_unroll(xs, h0, c0, p)[0]

    jac = central_difference_jacobian(fwd_x0, x_seq[0:di], dh)
    assert temporal_dependence(jac, dh, di) > 1e-6  # x_0 genuinely propagates to h_T


def test_lstm_params_validate():
    for kwargs in [{"input_dim": 0}, {"hidden_dim": 0}]:
        try:
            di, dh = kwargs.get("input_dim", 2), kwargs.get("hidden_dim", 2)
            LstmParams(
                [0.0] * (dh * di),
                [0.0] * (dh * dh),
                [0.0] * dh,
                [0.0] * (dh * di),
                [0.0] * (dh * dh),
                [0.0] * dh,
                [0.0] * (dh * di),
                [0.0] * (dh * dh),
                [0.0] * dh,
                [0.0] * (dh * di),
                [0.0] * (dh * dh),
                [0.0] * dh,
                di,
                dh,
            )
            assert False, f"{kwargs} should raise"
        except ValueError:
            pass


# ============================================================================================================
# 4. TIER B.2 -- the GRU cell: hand-computed forward + analytic grads vs central FD + the unroll.
# ============================================================================================================


def _gru_params(rng, di, dh):
    def w(n):
        return _vec(rng, n)

    return GruParams(
        w(dh * di),
        w(dh * dh),
        w(dh),
        w(dh * di),
        w(dh * dh),
        w(dh),
        w(dh * di),
        w(dh * dh),
        w(dh),
        di,
        dh,
    )


def test_gru_forward_matches_a_hand_computed_example():
    # A 1x1 GRU with W_*=1, U_*=0, b_*=0, x=[0.5], h_prev=[0.3].
    p = GruParams([1.0], [0.0], [0.0], [1.0], [0.0], [0.0], [1.0], [0.0], [0.0], 1, 1)
    h = gru_cell_reference([0.5], [0.3], p)
    z = sigmoid(0.5)
    r = sigmoid(0.5)
    n = math.tanh(0.5 + r * 0.0)  # U_n h_prev = 0
    h_exp = (1.0 - z) * n + z * 0.3
    assert abs(h[0] - h_exp) < 1e-15, (h, h_exp)


def test_gru_analytic_gradients_match_central_finite_difference():
    rng = random.Random(0xE40E)
    di, dh = 3, 2
    p = _gru_params(rng, di, dh)
    x, h_prev = _vec(rng, di), _vec(rng, dh)
    G = gru_cell_grads(x, h_prev, p)
    fd_dx = central_difference_jacobian(lambda v: gru_cell_reference(v, h_prev, p), x, dh)
    fd_dh = central_difference_jacobian(lambda v: gru_cell_reference(x, v, p), h_prev, dh)
    assert max_jacobian_error(G["dh_dx"], fd_dx) < 1e-6, max_jacobian_error(G["dh_dx"], fd_dx)
    assert max_jacobian_error(G["dh_dh_prev"], fd_dh) < 1e-6, max_jacobian_error(
        G["dh_dh_prev"], fd_dh
    )


def test_gru_unroll_and_temporal_dependence():
    rng = random.Random(0xE40F)
    di, dh, T = 2, 3, 5
    p = _gru_params(rng, di, dh)
    x_seq, h0 = _vec(rng, T * di), _vec(rng, dh)
    # unroll == repeated cell.
    h = list(h0)
    for t in range(T):
        h = gru_cell_reference(x_seq[t * di : (t + 1) * di], h, p)
    assert all(abs(a - b) < 1e-15 for a, b in zip(h, gru_unroll(x_seq, h0, p)))
    assert len(gru_states(x_seq, h0, p)) == T + 1
    # temporal dependence: h_T depends on x_0.
    jac = central_difference_jacobian(
        lambda x0: gru_unroll(x0 + x_seq[di:], h0, p), x_seq[0:di], dh
    )
    assert temporal_dependence(jac, dh, di) > 1e-6


def test_gru_is_side_effect_free():
    rng = random.Random(0xE410)
    di, dh = 3, 2
    p = _gru_params(rng, di, dh)
    x, h_prev = _vec(rng, di), _vec(rng, dh)
    x0, h0 = list(x), list(h_prev)
    gru_cell_reference(x, h_prev, p)
    assert x == x0 and h_prev == h0


# ============================================================================================================
# 5. The Q8 bridge: tracks the reference within the R17 input bound + is a clean round-trip.
# ============================================================================================================


def test_lstm_bridge_tracks_the_reference_within_quant_error():
    from bcir.kbcir.quantize import max_abs_error

    rng = random.Random(0xE411)
    di, dh, T = 4, 3, 4
    p = _lstm_params(rng, di, dh)
    x_seq, h0, c0 = _vec(rng, T * di, -1, 1), _vec(rng, dh), _vec(rng, dh)
    q_err = max_abs_error(x_seq, group_size=di, bits=8)
    h_ref, c_ref = lstm_unroll(x_seq, h0, c0, p)
    h_got, c_got = lstm_via_bridge(x_seq, h0, c0, p, group_size=di, bits=8)
    # the cell is a bounded smooth map (sigmoid/tanh are 1-Lipschitz-ish); a conservative multiple of the
    # per-element round-trip error covers the T-step amplification.
    assert all(abs(r - g) <= 100.0 * q_err + 1e-2 for r, g in zip(h_ref, h_got)), q_err
    assert all(abs(r - g) <= 100.0 * q_err + 1e-2 for r, g in zip(c_ref, c_got)), q_err


def test_lstm_bridge_is_unroll_of_the_roundtrip():
    from bcir.kbcir.quantize import dequantize, quantize_per_group

    rng = random.Random(0xE412)
    di, dh, T = 4, 2, 3
    p = _lstm_params(rng, di, dh)
    x_seq, h0, c0 = _vec(rng, T * di, -1, 1), _vec(rng, dh), _vec(rng, dh)
    dx = dequantize(quantize_per_group(x_seq, di, 8))
    assert lstm_via_bridge(x_seq, h0, c0, p, di, 8) == lstm_unroll(dx, h0, c0, p)


def test_gru_bridge_is_unroll_of_the_roundtrip():
    from bcir.kbcir.quantize import dequantize, quantize_per_group

    rng = random.Random(0xE413)
    di, dh, T = 4, 2, 3
    p = _gru_params(rng, di, dh)
    x_seq, h0 = _vec(rng, T * di, -1, 1), _vec(rng, dh)
    dx = dequantize(quantize_per_group(x_seq, di, 8))
    assert gru_via_bridge(x_seq, h0, p, di, 8) == gru_unroll(dx, h0, p)


def test_r17_bound_is_the_input_roundtrip():
    assert quantization_error_bound() == 1  # 1 ULP, as every wrap


# ============================================================================================================
# 6. Shape/dtype well-formedness (op-level validation, NOT a new global R-law).
# ============================================================================================================


def test_check_recurrent_accepts_well_formed_claims():
    assert check_recurrent("rnn", 3, 4, 5, "f32", "f32") == []
    assert check_recurrent("lstm", 8, 16, 10, "f32", "f32") == []
    assert check_recurrent("gru", 2, 2, 1, "f32", "f32") == []
    assert check_recurrent("rnn", 3, 4, 5, "i32", "i32") == []  # the closed-set relu-RNN may be i32


def test_check_recurrent_rejects_malformed_claims():
    assert any("not a known kind" in e for e in check_recurrent("foo", 3, 4, 5, "f32", "f32"))
    assert any(">= 1" in e for e in check_recurrent("rnn", 0, 4, 5, "f32", "f32"))
    assert any(">= 1" in e for e in check_recurrent("rnn", 3, 4, 0, "f32", "f32"))
    assert any("not preserved" in e for e in check_recurrent("lstm", 3, 4, 5, "f32", "i32"))
    # the quarantine dtype rule: LSTM/GRU's tanh/sigmoid need f32.
    assert any("needs f32" in e for e in check_recurrent("lstm", 3, 4, 5, "i32", "i32"))
    assert any("needs f32" in e for e in check_recurrent("gru", 3, 4, 5, "i32", "i32"))
    assert any("not a known dtype" in e for e in check_recurrent("rnn", 3, 4, 5, "f64", "f64"))


# ============================================================================================================
# 7. The emitted LSTM C kernel compiles + reproduces the reference (self-skip when no toolchain).
# ============================================================================================================


def _cc():
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def _arr(v):
    return "{" + ", ".join(f"{f:.7f}f" for f in v) + "}"


def _compile_run_lstm(x, h_prev, c_prev, p):
    cc = _cc()
    di, dh = p.input_dim, p.hidden_dim
    kernel = emit_lstm_cell_c(di, dh, "lc")
    main = (
        f"\n#include <stdio.h>\nint main(void){{\n"
        f"  float X[{di}]={_arr(x)}; float HP[{dh}]={_arr(h_prev)}; float CP[{dh}]={_arr(c_prev)};\n"
        f"  float Wf[{dh * di}]={_arr(p.W_f)}, Uf[{dh * dh}]={_arr(p.U_f)}, bf[{dh}]={_arr(p.b_f)};\n"
        f"  float Wi[{dh * di}]={_arr(p.W_i)}, Ui[{dh * dh}]={_arr(p.U_i)}, bi[{dh}]={_arr(p.b_i)};\n"
        f"  float Wo[{dh * di}]={_arr(p.W_o)}, Uo[{dh * dh}]={_arr(p.U_o)}, bo[{dh}]={_arr(p.b_o)};\n"
        f"  float Wg[{dh * di}]={_arr(p.W_g)}, Ug[{dh * dh}]={_arr(p.U_g)}, bg[{dh}]={_arr(p.b_g)};\n"
        f"  float H[{dh}], C[{dh}];\n"
        f"  lc(X,HP,CP,Wf,Uf,bf,Wi,Ui,bi,Wo,Uo,bo,Wg,Ug,bg,H,C);\n"
        f'  for (int i=0;i<{dh};++i) printf("%.7f ", H[i]);\n'
        f'  for (int i=0;i<{dh};++i) printf("%.7f ", C[i]);\n  return 0;\n}}\n'
    )
    with tempfile.TemporaryDirectory() as dd:
        src = os.path.join(dd, "lc.c")
        open(src, "w").write(kernel + main)
        exe = os.path.join(dd, "lc")
        bld = subprocess.run(
            host_link_args([cc, "-std=c11", "-O2", src, "-lm", "-o", exe]),
            capture_output=True,
            text=True,
        )
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        return [float(t) for t in out.stdout.split()]


def test_lstm_c_kernel_matches_the_reference():
    cc = _cc()
    if not cc:
        return  # quick tier hides the toolchain -> self-skip
    rng = random.Random(0xE414)
    di, dh = 3, 2
    p = _lstm_params(rng, di, dh)
    x, h_prev, c_prev = _vec(rng, di), _vec(rng, dh), _vec(rng, dh)
    got = _compile_run_lstm(x, h_prev, c_prev, p)
    h_ref, c_ref = lstm_cell_reference(x, h_prev, c_prev, p)
    ref = h_ref + c_ref
    # the C kernel calls the SAME libm tanhf/expf; float32 vs the oracle's float64 differ by round-off only.
    assert all(abs(g - r) < 1e-3 for g, r in zip(got, ref)), (got, ref)


def test_lstm_c_emit_routes_transcendentals_through_the_libm_edge():
    src = emit_lstm_cell_c(3, 2, "lc")
    assert (
        "math.h" in src and "tanhf" in src and "expf" in src
    )  # the gate transcendentals ride the libm edge
    assert "c.call.libm:" in src  # the edge label in the docstring comment
    assert "void lc(const float *restrict x," in src  # the baked-in signature
    for bad in [(0, 2), (2, 0)]:
        try:
            emit_lstm_cell_c(*bad)
            assert False, f"{bad} should raise"
        except ValueError:
            pass


# ============================================================================================================
# 8. The link-flag rule: tanhf/expf -> -lm (REUSES the existing libm rule; no linkflags change).
# ============================================================================================================


def test_recurrent_link_flag_rule_reuses_the_existing_libm_rule():
    # The LSTM/GRU gate tanhf + the sigmoid's expf MATCH the existing libm rule -- NO linkflags change needed.
    assert library_for_callee("tanhf") == "-lm"  # the gate tanh edge
    assert library_for_callee("expf") == "-lm"  # the sigmoid exp edge
    assert library_for_callee("tanh") == "-lm"  # the double forms too
    assert library_for_callee("exp") == "-lm"
    # no regression on the other library rules + libc + unknown.
    assert library_for_callee("sqrtf") == "-lm"  # E3 layernorm still maps
    assert library_for_callee("LAPACKE_ssyev") == "-llapack"  # E2 PCA still maps
    assert library_for_callee("cblas_sgemm") == "-lcblas"  # B5 BLAS still maps
    assert library_for_callee("free") == NO_FLAG  # libc-implicit, known
    assert library_for_callee("totally_unknown_fn") is None  # unknown-callee policy unchanged


# ============================================================================================================
# the two-truth quarantine: the recurrent module imports no verifier / emits no Diagnostic.
# ============================================================================================================


def test_recurrent_touches_no_verifier_and_emits_no_diagnostic():
    # The recurrent cells are cost-side / oracle quantities (off the legality path), never an R-law verdict.
    # AST-inspect the module: no verify / Diagnostic / Graded names bound, no verifier module imported (exactly
    # as test_ols / test_pca / test_transformer).
    import bcir.kbcir.recurrent as rc_mod

    bound = set(vars(rc_mod))
    assert not (bound & {"verify_quarantine", "Diagnostic", "Graded", "decide"}), sorted(bound)
    tree = ast.parse(open(rc_mod.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("verify" in m for m in imported), sorted(imported)
    # the returned values are plain floats / plain node ids (no graded/confidence wrapper).
    rng = random.Random(0xE415)
    p = _lstm_params(rng, 2, 2)
    h, c = lstm_cell_reference(_vec(rng, 2), _vec(rng, 2), _vec(rng, 2), p)
    assert isinstance(h, list) and all(isinstance(v, float) for v in h)
    assert isinstance(c, list) and all(isinstance(v, float) for v in c)


def test_tier_b_lstm_readout_training_drives_loss_down():
    # E4 Tier-B CONVERGENCE (H4): Tier-A (relu-RNN) already has a training-convergence test; Tier-B (LSTM) had
    # only forward + analytic-gradient checks. The LSTM forward is the (fixed) feature extractor -- its final
    # hidden state -- and the M3 loop fits a linear readout, MSE dropping to ~0. The transcendental gates
    # (sigmoid / tanh) break the closed-set autodiff closure, so they are frozen OUTSIDE the Tape; only the
    # closed-set readout trains, exactly as the Tier-A convergence test does. Gives Tier-B a training gate.
    from bcir.kbcir.training import Dataset, train

    rng = random.Random(0xE4C0)
    di, dh, T = 2, 3, 4
    p = _lstm_params(rng, di, dh)
    coef = [0.6, -0.4, 0.25]
    feats, ys = [], []
    for _ in range(24):
        seq = _vec(rng, T * di, -0.8, 0.8)
        hT, _cT = lstm_unroll(seq, [0.0] * dh, [0.0] * dh, p)
        feats.append(tuple(hT))
        ys.append(sum(coef[j] * hT[j] for j in range(dh)) + 0.05)

    def model(tape, names, X):
        r = [tape.var(nm) for nm in names]
        out = []
        for row in X:
            pred = r[dh]
            for j in range(dh):
                pred = tape.add(pred, tape.mul(r[j], tape.const(row[j])))
            out.append(pred)
        return out

    names = [f"r{j}" for j in range(dh)] + ["b"]
    res = train(
        model,
        [0.0] * (dh + 1),
        Dataset(tuple(feats), tuple(ys)),
        loss="mse",
        optimizer="adam",
        epochs=120,
        lr=0.05,
        seed=1,
        param_names=names,
    )
    # the shared convergence gate: decreased + absolute threshold + substantial drop.
    assert_converged(res.train_loss, final_below=1e-2, name="E4 Tier-B LSTM readout")


def test_tier_b_gru_readout_training_drives_loss_down():
    # E4 Tier-B GRU CONVERGENCE: the GRU previously had forward + analytic-gradient evidence but NO training
    # gate of any kind (the audit gap). Exactly the LSTM pattern: the GRU forward is the (fixed) feature
    # extractor -- its final hidden state -- and the M3 loop fits a linear readout, MSE dropping to ~0. The
    # transcendental gates are frozen OUTSIDE the Tape; only the closed-set readout trains.
    from bcir.kbcir.training import Dataset, train

    rng = random.Random(0xE4C1)
    di, dh, T = 2, 3, 4
    p = _gru_params(rng, di, dh)
    coef = [0.5, -0.35, 0.2]
    feats, ys = [], []
    for _ in range(24):
        seq = _vec(rng, T * di, -0.8, 0.8)
        hT = gru_unroll(seq, [0.0] * dh, p)
        feats.append(tuple(hT))
        ys.append(sum(coef[j] * hT[j] for j in range(dh)) + 0.05)

    def model(tape, names, X):
        r = [tape.var(nm) for nm in names]
        out = []
        for row in X:
            pred = r[dh]
            for j in range(dh):
                pred = tape.add(pred, tape.mul(r[j], tape.const(row[j])))
            out.append(pred)
        return out

    names = [f"r{j}" for j in range(dh)] + ["b"]
    res = train(
        model,
        [0.0] * (dh + 1),
        Dataset(tuple(feats), tuple(ys)),
        loss="mse",
        optimizer="adam",
        epochs=120,
        lr=0.05,
        seed=1,
        param_names=names,
    )
    assert_converged(res.train_loss, final_below=1e-2, name="E4 Tier-B GRU readout")


def test_tier_a_rnn_weights_train_end_to_end():
    # E4 Tier-A WEIGHT training (the audit gap): the BPTT weight gradients were FD-verified
    # (test_tier_a_bptt_matches_finite_difference_wrt_weights) but never consumed by an optimizer -- the
    # readout convergence test bakes the RNN weights as tape.const. Here the RNN weights THEMSELVES are
    # tape.var: labels come from a teacher relu-RNN under a FIXED sum readout, and a differently-initialized
    # student trains its wx/wh/b through the unrolled DAG (literal BPTT through time) until the loss
    # collapses -- the recurrent weights, not a readout, do the learning. Positive inputs + positive biases
    # keep every unit off the relu kink, exactly as the headline FD tests do.
    from bcir.kbcir.training import Dataset, train

    rng = random.Random(0xE4C2)
    di, dh, T = 1, 2, 3
    teacher = RnnParams(
        _vec(rng, dh * di, 0.3, 0.7), _vec(rng, dh * dh, 0.1, 0.3), _vec(rng, dh, 0.1, 0.3), di, dh
    )
    seqs, ys = [], []
    for _ in range(24):
        s = [rng.uniform(0.4, 1.0) for _ in range(T * di)]
        hT = rnn_relu_reference(s, [0.0] * dh, teacher)
        seqs.append(tuple(s))
        ys.append(hT[0] + hT[1])  # the FIXED readout: sum of the final state

    names = (
        [f"wx{i}" for i in range(dh * di)]
        + [f"wh{i}" for i in range(dh * dh)]
        + [f"b{i}" for i in range(dh)]
    )

    def model(tape, param_names, X):
        w = {nm: tape.var(nm) for nm in param_names}
        zero = tape.const(0.0)
        out = []
        for row in X:
            h = [tape.const(0.0) for _ in range(dh)]
            for t in range(T):
                xc = [tape.const(row[t * di + j]) for j in range(di)]
                new_h = []
                for o in range(dh):
                    z = w[f"b{o}"]
                    for j in range(di):
                        z = tape.add(z, tape.mul(w[f"wx{o * di + j}"], xc[j]))
                    for k in range(dh):
                        z = tape.add(z, tape.mul(w[f"wh{o * dh + k}"], h[k]))
                    new_h.append(tape.select(z, z, zero))  # relu (closed-set)
                h = new_h
            out.append(tape.add(h[0], h[1]))  # fixed sum readout: no readout params
        return out

    p0 = [rng.uniform(0.35, 0.65) for _ in names]  # a different (positive) init from the teacher
    res = train(
        model,
        p0,
        Dataset(tuple(seqs), tuple(ys)),
        loss="mse",
        optimizer="adam",
        epochs=200,
        batch_size=8,
        lr=0.05,
        seed=1,
        param_names=names,
    )
    assert list(res.params) != list(p0)  # the gradient genuinely flowed into the weights
    assert_converged(res.train_loss, final_below=1e-3, max_ratio=0.1, name="E4 Tier-A RNN weights")


if __name__ == "__main__":
    import sys

    mod = sys.modules[__name__]
    tests = sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n)))
    passed = failed = 0
    for name in tests:
        try:
            getattr(mod, name)()
            passed += 1
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {e!r}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
