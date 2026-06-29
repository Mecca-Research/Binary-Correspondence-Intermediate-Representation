"""E4 (ML-breadth) -- RECURRENT CELLS (RNN / LSTM / GRU): a TWO-TIER oracle, the FOURTH slice of the ML-breadth
ladder after E1 (OLS), E2 (PCA), E3 (the full Transformer block).

THE TWO-TIER DESIGN -- forced by the B3 autodiff's CLOSED primitive set (this exactly mirrors M1's two-path
losses, ``bcir/kbcir/losses.py``). ``bcir.kbcir.autodiff``'s ``Tape`` has a CLOSED differentiable primitive set
``{const, var, neg, add, sub, mul, div, dot, select}`` -- there are NO transcendentals (no tanh / sigmoid / exp).
``select(cond, a, b) = a if cond > 0 else b``, so ``relu(x) = select(x, x, const(0))`` IS expressible in the
closed set, but ``tanh`` / ``sigmoid`` are NOT. So a recurrent cell splits into two tiers exactly as a loss does
(closed-set MSE built into the Tape vs. a transcendental loss that supplies a closed-form gradient SEED):

  * TIER A -- the CLOSED-SET relu-RNN, trainable END-TO-END via the EXISTING ``unroll_scan`` + ``grad`` (= literal
    BPTT). An Elman RNN with a relu activation ``h_t = relu(W_h h_{t-1} + W_x x_t + b)`` is built from ONLY closed
    primitives: the two matmuls via ``dot``, the bias add via ``add``, the relu via ``select(z, z, const(0))`` per
    element. Folding the step over a sequence (``unroll_scan`` -- here an explicit Python unroll over a VECTOR
    carry, see ``rnn_relu_unroll``) yields one output DAG; running the EXISTING ``autodiff.grad`` over a scalar
    readout of the final state IS backprop-through-time. The ``unroll_scan`` docstring states this exactly:
    ``d(final_carry)/d(input) == reverse-mode AD over the unrolled DAG == BPTT``. THE HEADLINE GATE verifies the
    BPTT gradient against ``autodiff.finite_difference_grad`` via the existing ``autodiff.gradients_match`` -- a
    real trainable recurrent net whose BPTT is the existing reverse-mode AD, checked by finite differences.

  * TIER B -- LSTM and GRU cells (transcendental ``tanh`` / ``sigmoid`` gates), the M1 closed-form-SEED treatment.
    A NUMERIC forward reference (the standard cell math via ``math.tanh`` / a guarded sigmoid -- these ride the
    trusted libm edge, OFF the legality rail, exactly as ``activation.py`` / ``losses.py`` do for the
    transcendental path), plus CLOSED-FORM analytic gradients built from the gate derivatives
    ``sigma'(x) = sigma(x)(1 - sigma(x))`` and ``tanh'(x) = 1 - tanh(x)^2`` -- the seed, exactly like
    ``losses.py``'s ``*_grad_logits``. The forward is checked against a hand-computed reference; the analytic
    gradient against CENTRAL finite differences (numeric, since tanh/sigmoid are NOT closed Tape primitives so the
    closed Tape autodiff cannot differentiate them).

WHY TIER A IS AN EXPLICIT VECTOR UNROLL (not a single ``unroll_scan`` call). ``unroll_scan(tape, step, init, xs)``
carries a SINGLE node id as the recurrence carry, but an RNN's hidden state is a VECTOR. The honest reading of the
``unroll_scan`` contract is that a bounded scan is *"just a finite composition of primitives"* -- so we unroll the
vector recurrence with an explicit Python ``for`` loop that calls the per-step builder, building exactly the same
finite composition of closed-primitive nodes ``unroll_scan`` would for a scalar carry. The result is one output
DAG whose ``grad`` is BPTT, with no new machinery -- the ``unroll_scan`` docstring's contract is cited, not
bypassed (a single-hidden-unit RNN is run through ``unroll_scan`` VERBATIM too, see ``rnn_scalar_step``, to pin
that the explicit unroll is the same composition the scan builds).

THE QUARANTINE BOUNDARY (mirrors losses.py / activation.py / transformer.py). Tier A is fully on the closed
deterministic rail (relu via ``select`` -- exact, re-differentiable). Tier B's ``tanh`` / ``sigmoid`` are the only
transcendentals; they ride the trusted ``c.call.libm:`` edge (``tanhf`` / ``expf`` -> ``-lm``, which the cfront
link rail ALREADY maps: ``library_for_callee("tanhf") == library_for_callee("expf") == "-lm"`` -- NO linkflags
change). The numeric forward references match a from-scratch composition to float round-off; the analytic cell
gradient is the closed-form seed, checked against central finite differences. The Q8 bridge round-trips the INPUT
sequence then runs the trusted forward, so the SOLE certified error is the R17 input bound -- exactly as
``transformer_block_via_bridge`` / ``attention_via_bridge``.

LAYOUT (row-major / flat throughout). A sequence ``x_seq`` is ``T`` time-steps of an ``input_dim`` vector, flat
row-major: ``x_seq[t*input_dim + j]`` is time-step t's feature j. A weight ``W_*`` mapping a ``d_in`` vector to a
``d_out`` vector is ``d_out x d_in`` row-major (``W[o*d_in + i]``); a recurrent weight ``U_*`` is
``hidden_dim x hidden_dim``; a bias ``b_*`` is length ``d_out``. All references return FRESH buffers and do NOT
mutate their inputs (side-effect-free oracle).

This module is OFF the legality path: it imports NO verifier, emits NO ``Diagnostic``, and returns plain floats /
plain node ids (no graded/confidence wrapper) -- exactly like ``losses.py`` / ``transformer.py`` / ``pca.py`` /
``ols.py``. Deterministic, dependency-free (stdlib ``math`` only). A research/oracle organ (kept cold).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .autodiff import Tape
from .quantize import dequantize, quantize_per_group


# ============================================================================================================
# Tier A -- the CLOSED-SET relu-RNN: a step builder over the autodiff Tape, an explicit vector unroll = BPTT.
# ============================================================================================================

@dataclass(frozen=True)
class RnnParams:
    """The weights of one Elman (vanilla) RNN cell, all flat row-major. ``w_x`` is the input weight
    ``hidden_dim x input_dim`` (``w_x[h*input_dim + j]``); ``w_h`` is the recurrent weight
    ``hidden_dim x hidden_dim`` (``w_h[h*hidden_dim + k]``); ``b`` is the length-``hidden_dim`` bias. The same
    carrier feeds both the Tape-built closed-set path (``rnn_relu_step``) and the plain-float reference
    (``rnn_relu_reference``), so the two are guaranteed to be the SAME cell."""

    w_x: list[float]
    w_h: list[float]
    b: list[float]
    input_dim: int
    hidden_dim: int

    def __post_init__(self) -> None:
        di, dh = self.input_dim, self.hidden_dim
        if di < 1 or dh < 1:
            raise ValueError(f"rnn dims must be >= 1; got input_dim={di} hidden_dim={dh}")
        if len(self.w_x) != dh * di:
            raise ValueError(f"rnn w_x must be hidden_dim*input_dim = {dh * di}; got {len(self.w_x)}")
        if len(self.w_h) != dh * dh:
            raise ValueError(f"rnn w_h must be hidden_dim*hidden_dim = {dh * dh}; got {len(self.w_h)}")
        if len(self.b) != dh:
            raise ValueError(f"rnn b must be length hidden_dim = {dh}; got {len(self.b)}")


def rnn_relu_step(tape: Tape, h_prev_ids: tuple, x_ids: tuple, params: RnnParams) -> tuple:
    """Build ONE step of the Elman relu-RNN as CLOSED-SET Tape nodes and return the new hidden-state node-id
    tuple (length ``hidden_dim``):

        z_h = dot(W_h[h], h_prev) + dot(W_x[h], x) + b[h]        (the affine pre-activation per unit h)
        h_t[h] = relu(z_h) = select(z_h, z_h, const(0))          (relu via the differentiable select)

    Every operation is a Tape primitive: ``dot`` for each matmul row (the matmul-substrate primitive), ``add``
    for the sum + bias, and ``select(z, z, const(0))`` for the per-element relu (relu = max(0, z), exactly the
    autodiff ``select`` shape -- the gradient flows through the closed set). ``h_prev_ids`` / ``x_ids`` are
    tuples of Tape node ids (the previous hidden state and this step's input); the weights come from ``params``
    as ``const`` nodes. Because the result is rooted in the SAME closed primitive set, the EXISTING
    ``autodiff.grad`` differentiates the WHOLE unrolled recurrence end-to-end -- this step builder is the body a
    scan/unroll folds, and reverse mode over the fold is literally BPTT (see ``rnn_relu_unroll``).

    NOTE this is the per-step body; ``rnn_relu_unroll`` is the explicit vector unroll that folds it over time
    (an ``unroll_scan`` over a VECTOR carry -- the same finite composition of primitives the scalar
    ``unroll_scan`` builds, see the module docstring). Returns a tuple of ``hidden_dim`` node ids."""
    di, dh = params.input_dim, params.hidden_dim
    if len(h_prev_ids) != dh:
        raise ValueError(f"rnn_relu_step: h_prev must have hidden_dim = {dh} nodes; got {len(h_prev_ids)}")
    if len(x_ids) != di:
        raise ValueError(f"rnn_relu_step: x must have input_dim = {di} nodes; got {len(x_ids)}")
    zero = tape.const(0.0)
    new_h: list[int] = []
    for h in range(dh):
        # z_h = W_x[h] . x + W_h[h] . h_prev + b[h], all closed-set nodes.
        wx_row = tuple(tape.const(params.w_x[h * di + j]) for j in range(di))
        wh_row = tuple(tape.const(params.w_h[h * dh + k]) for k in range(dh))
        z = tape.add(tape.dot(wx_row, tuple(x_ids)), tape.dot(wh_row, tuple(h_prev_ids)))
        z = tape.add(z, tape.const(params.b[h]))
        # relu(z) = max(0, z) = select(z, z, 0): cond=z>0 picks z, else 0 -- the differentiable a.e. relu.
        new_h.append(tape.select(z, z, zero))
    return tuple(new_h)


def rnn_scalar_step(tape: Tape, carry_nid: int, x_nid: int, *, w_h: float, w_x: float, b: float) -> int:
    """A SINGLE-hidden-unit (scalar-carry) relu-RNN step in the EXACT ``unroll_scan`` step signature
    ``step(tape, carry_nid, x_nid) -> nid``: ``h_t = relu(w_h*carry + w_x*x + b)``. Provided so the scalar
    recurrence can be folded by ``autodiff.unroll_scan`` VERBATIM (``rnn_relu_unroll_scalar``), pinning that the
    explicit vector unroll in ``rnn_relu_unroll`` is the SAME finite composition the scan builds for a scalar
    carry -- the contract ``unroll_scan`` already differentiates and the FD gate already checks."""
    z = tape.add(tape.add(tape.mul(tape.const(w_h), carry_nid), tape.mul(tape.const(w_x), x_nid)),
                 tape.const(b))
    zero = tape.const(0.0)
    return tape.select(z, z, zero)                          # relu via the differentiable select


def rnn_relu_unroll(tape: Tape, x_var_names: list, h0_ids: tuple, params: RnnParams) -> tuple:
    """Unroll the Elman relu-RNN over a sequence by folding ``rnn_relu_step`` time-step by time-step (the
    explicit VECTOR-carry unroll = ``unroll_scan`` over a vector state -- the same finite composition of closed
    primitives the scalar ``unroll_scan`` builds; see the module docstring). Returns the FINAL hidden state as a
    tuple of ``hidden_dim`` node ids.

    ``x_var_names`` is a length-``T`` list whose t-th entry is a length-``input_dim`` list of VAR NAMES (so the
    inputs are differentiation targets -- the BPTT ``d(h_T)/d(x_t)`` gate differentiates w.r.t. them); the
    builder binds each name with ``tape.var``. ``h0_ids`` is the initial hidden state's node-id tuple. The
    output DAG is rooted in the closed primitive set, so ``autodiff.grad`` over a scalar readout of the returned
    state IS backprop-through-time over the whole sequence -- no extra machinery."""
    di, dh = params.input_dim, params.hidden_dim
    if len(h0_ids) != dh:
        raise ValueError(f"rnn_relu_unroll: h0 must have hidden_dim = {dh} nodes; got {len(h0_ids)}")
    h = tuple(h0_ids)
    for t, names in enumerate(x_var_names):
        if len(names) != di:
            raise ValueError(f"rnn_relu_unroll: x[{t}] must have input_dim = {di} names; got {len(names)}")
        x_ids = tuple(tape.var(nm) for nm in names)
        h = rnn_relu_step(tape, h, x_ids, params)
    return h


def rnn_relu_reference(x_seq: list[float], h0: list[float], params: RnnParams) -> list[float]:
    """Plain-float forward of the Elman relu-RNN over a sequence (NOT via the Tape) -- the independent
    cross-check for ``evaluate`` over the unrolled DAG. ``x_seq`` is ``T x input_dim`` flat row-major
    (``x_seq[t*input_dim + j]``); ``h0`` is the length-``hidden_dim`` initial state. Returns the FINAL hidden
    state (length ``hidden_dim``). Recurrence: ``h_t = relu(W_h h_{t-1} + W_x x_t + b)``. Side-effect-free."""
    di, dh = params.input_dim, params.hidden_dim
    if len(h0) != dh:
        raise ValueError(f"rnn_relu_reference: h0 must be length hidden_dim = {dh}; got {len(h0)}")
    if len(x_seq) % di != 0:
        raise ValueError(f"rnn_relu_reference: x_seq length {len(x_seq)} not a multiple of input_dim = {di}")
    T = len(x_seq) // di
    h = [float(v) for v in h0]
    for t in range(T):
        xt = x_seq[t * di:(t + 1) * di]
        new_h = [0.0] * dh
        for o in range(dh):
            z = params.b[o]
            for j in range(di):
                z += params.w_x[o * di + j] * float(xt[j])
            for k in range(dh):
                z += params.w_h[o * dh + k] * h[k]
            new_h[o] = z if z > 0.0 else 0.0               # relu (a.e.; same convention as the select VJP)
        h = new_h
    return h


def rnn_relu_states(x_seq: list[float], h0: list[float], params: RnnParams) -> list[list[float]]:
    """Like :func:`rnn_relu_reference` but returns EVERY hidden state ``[h_0, h_1, ..., h_T]`` (length ``T+1``,
    each a length-``hidden_dim`` list) -- used by the temporal-dependence verifier (does ``h_T`` depend on the
    early inputs?) and to inspect the full trajectory. Side-effect-free."""
    di, dh = params.input_dim, params.hidden_dim
    T = len(x_seq) // di
    h = [float(v) for v in h0]
    traj = [list(h)]
    for t in range(T):
        xt = x_seq[t * di:(t + 1) * di]
        new_h = [0.0] * dh
        for o in range(dh):
            z = params.b[o]
            for j in range(di):
                z += params.w_x[o * di + j] * float(xt[j])
            for k in range(dh):
                z += params.w_h[o * dh + k] * h[k]
            new_h[o] = z if z > 0.0 else 0.0
        h = new_h
        traj.append(list(h))
    return traj


# ============================================================================================================
# Tier B -- the transcendental gates: sigmoid / tanh helpers + their closed-form derivatives (the seed).
# ============================================================================================================

def sigmoid(x: float) -> float:
    """The logistic sigmoid ``1 / (1 + exp(-x))`` -- numerically guarded for large ``|x|`` so no OverflowError.
    For ``x >= 0`` use ``1/(1+exp(-x))`` directly (``exp(-x) <= 1``, no overflow); for ``x < 0`` use the
    algebraically-equal ``exp(x)/(1+exp(x))`` (``exp(x) < 1``, no overflow). A transcendental (rides the libm
    ``expf`` edge in the C twin) -- it is the gate value only; its DERIVATIVE is the closed-form seed below."""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def sigmoid_prime_from_value(s: float) -> float:
    """The sigmoid derivative expressed in terms of the sigmoid VALUE ``s = sigmoid(x)``:
    ``sigma'(x) = s * (1 - s)`` -- the closed-form gate derivative (the M1-style seed). Given the already-computed
    forward sigmoid it needs NO new transcendental, so it lives entirely in the closed set; this is exactly the
    fused-gradient trick the LSTM/GRU backward uses."""
    return s * (1.0 - s)


def tanh_prime_from_value(t: float) -> float:
    """The tanh derivative in terms of the tanh VALUE ``t = tanh(x)``: ``tanh'(x) = 1 - t^2`` -- the closed-form
    gate derivative (the seed). Like the sigmoid one, it reuses the forward value and adds no transcendental."""
    return 1.0 - t * t


def _row_dot(w: list[float], v: list[float], o: int, d_in: int) -> float:
    """Row ``o`` of a ``d_out x d_in`` row-major weight ``w`` dotted with the length-``d_in`` vector ``v``
    (``sum_i w[o*d_in + i] * v[i]``) -- the scalar matmul row the cell gates use. Exact/deterministic."""
    acc = 0.0
    for i in range(d_in):
        acc += float(w[o * d_in + i]) * float(v[i])
    return acc


# ============================================================================================================
# Tier B.1 -- the LSTM cell: numeric forward reference + the closed-form analytic gradients (the seed).
# ============================================================================================================

@dataclass(frozen=True)
class LstmParams:
    """The weights of one LSTM cell, all flat row-major. For each gate g in {f (forget), i (input), o (output),
    g (candidate)}: ``W_g`` is the input weight ``hidden_dim x input_dim``, ``U_g`` the recurrent weight
    ``hidden_dim x hidden_dim``, ``b_g`` the length-``hidden_dim`` bias. ``input_dim`` / ``hidden_dim`` are the
    extents. Validated in ``__post_init__`` (positive dims, consistent weight/bias shapes)."""

    W_f: list[float]; U_f: list[float]; b_f: list[float]
    W_i: list[float]; U_i: list[float]; b_i: list[float]
    W_o: list[float]; U_o: list[float]; b_o: list[float]
    W_g: list[float]; U_g: list[float]; b_g: list[float]
    input_dim: int
    hidden_dim: int

    def __post_init__(self) -> None:
        di, dh = self.input_dim, self.hidden_dim
        if di < 1 or dh < 1:
            raise ValueError(f"lstm dims must be >= 1; got input_dim={di} hidden_dim={dh}")
        for nm, W in (("W_f", self.W_f), ("W_i", self.W_i), ("W_o", self.W_o), ("W_g", self.W_g)):
            if len(W) != dh * di:
                raise ValueError(f"lstm {nm} must be hidden_dim*input_dim = {dh * di}; got {len(W)}")
        for nm, U in (("U_f", self.U_f), ("U_i", self.U_i), ("U_o", self.U_o), ("U_g", self.U_g)):
            if len(U) != dh * dh:
                raise ValueError(f"lstm {nm} must be hidden_dim*hidden_dim = {dh * dh}; got {len(U)}")
        for nm, b in (("b_f", self.b_f), ("b_i", self.b_i), ("b_o", self.b_o), ("b_g", self.b_g)):
            if len(b) != dh:
                raise ValueError(f"lstm {nm} must be length hidden_dim = {dh}; got {len(b)}")


def lstm_cell_reference(x: list[float], h_prev: list[float], c_prev: list[float],
                        params: LstmParams) -> tuple[list[float], list[float]]:
    """ONE standard LSTM cell step (numeric, the source of truth). For each hidden unit, with gate
    pre-activations ``a_g = W_g x + U_g h_prev + b_g``:

        f = sigmoid(a_f)      (forget gate)        i = sigmoid(a_i)      (input gate)
        o = sigmoid(a_o)      (output gate)        g = tanh(a_g)         (candidate / cell input)
        c = f (.) c_prev + i (.) g                 (the new cell state)
        h = o (.) tanh(c)                          (the new hidden state)

    ``(.)`` is the elementwise product. Returns ``(h, c)`` -- each a fresh length-``hidden_dim`` list. The
    ``sigmoid`` / ``tanh`` are the transcendentals (ride the trusted libm edge); the matmuls / adds / products
    are exact. Side-effect-free (``x`` / ``h_prev`` / ``c_prev`` not mutated). Validates the input lengths."""
    di, dh = params.input_dim, params.hidden_dim
    if len(x) != di:
        raise ValueError(f"lstm_cell: x must be length input_dim = {di}; got {len(x)}")
    if len(h_prev) != dh:
        raise ValueError(f"lstm_cell: h_prev must be length hidden_dim = {dh}; got {len(h_prev)}")
    if len(c_prev) != dh:
        raise ValueError(f"lstm_cell: c_prev must be length hidden_dim = {dh}; got {len(c_prev)}")
    h = [0.0] * dh
    c = [0.0] * dh
    for u in range(dh):
        a_f = _row_dot(params.W_f, x, u, di) + _row_dot(params.U_f, h_prev, u, dh) + params.b_f[u]
        a_i = _row_dot(params.W_i, x, u, di) + _row_dot(params.U_i, h_prev, u, dh) + params.b_i[u]
        a_o = _row_dot(params.W_o, x, u, di) + _row_dot(params.U_o, h_prev, u, dh) + params.b_o[u]
        a_g = _row_dot(params.W_g, x, u, di) + _row_dot(params.U_g, h_prev, u, dh) + params.b_g[u]
        f, i, o, g = sigmoid(a_f), sigmoid(a_i), sigmoid(a_o), math.tanh(a_g)
        c[u] = f * float(c_prev[u]) + i * g
        h[u] = o * math.tanh(c[u])
    return h, c


def lstm_cell_grads(x: list[float], h_prev: list[float], c_prev: list[float],
                    params: LstmParams) -> dict:
    """The CLOSED-FORM analytic gradients of one LSTM cell's outputs w.r.t. its inputs (``x`` / ``h_prev`` /
    ``c_prev``), via the gate derivatives ``sigma' = s(1-s)`` and ``tanh' = 1 - t^2`` (the M1-style seed). The
    gradient is verified against CENTRAL finite differences (numeric, since tanh/sigmoid are not closed Tape
    primitives -- the closed Tape autodiff cannot differentiate them).

    Returns a dict of Jacobian blocks, each a flat ``hidden_dim x ?`` row-major list (``J[u*? + k]`` = the
    derivative of output unit ``u`` w.r.t. input element ``k``):
      ``dh_dx``      (``hidden_dim x input_dim``)   d h[u] / d x[j]
      ``dh_dh_prev`` (``hidden_dim x hidden_dim``)  d h[u] / d h_prev[k]
      ``dh_dc_prev`` (``hidden_dim x hidden_dim``)  d h[u] / d c_prev[k]
      ``dc_dx``      (``hidden_dim x input_dim``)   d c[u] / d x[j]
      ``dc_dh_prev`` (``hidden_dim x hidden_dim``)  d c[u] / d h_prev[k]
      ``dc_dc_prev`` (``hidden_dim x hidden_dim``)  d c[u] / d c_prev[k]

    Each hidden unit ``u`` is INDEPENDENT across the output index (the cell is elementwise in ``u`` given the
    gate pre-activations), but each gate pre-activation depends on the FULL ``x`` / ``h_prev`` vectors through
    its weight row -- so the per-input derivatives carry the weight entries. The chain, per unit u:

        c[u]      = f c_prev[u] + i g
        d c / d(.) = c_prev[u] * f'(a_f) * dA_f + f * d(c_prev[u])
                   + g * i'(a_i) * dA_i + i * g'(a_g) * dA_g
        h[u]      = o * tanh(c[u])
        d h / d(.) = tanh(c[u]) * o'(a_o) * dA_o + o * (1 - tanh(c[u])^2) * d c / d(.)

    where ``f' = f(1-f)``, ``i' = i(1-i)``, ``o' = o(1-o)``, ``g' = 1 - g^2`` are the closed-form gate
    derivatives, and ``dA_*`` is the gate pre-activation's derivative w.r.t. the chosen input (a weight entry
    for ``x`` / ``h_prev``, and 0 for ``c_prev`` since the gates do not see ``c_prev``). Side-effect-free."""
    di, dh = params.input_dim, params.hidden_dim
    if len(x) != di or len(h_prev) != dh or len(c_prev) != dh:
        raise ValueError("lstm_cell_grads: x/h_prev/c_prev length mismatch with params dims")
    dh_dx = [0.0] * (dh * di)
    dh_dh = [0.0] * (dh * dh)
    dh_dc = [0.0] * (dh * dh)
    dc_dx = [0.0] * (dh * di)
    dc_dh = [0.0] * (dh * dh)
    dc_dc = [0.0] * (dh * dh)
    for u in range(dh):
        a_f = _row_dot(params.W_f, x, u, di) + _row_dot(params.U_f, h_prev, u, dh) + params.b_f[u]
        a_i = _row_dot(params.W_i, x, u, di) + _row_dot(params.U_i, h_prev, u, dh) + params.b_i[u]
        a_o = _row_dot(params.W_o, x, u, di) + _row_dot(params.U_o, h_prev, u, dh) + params.b_o[u]
        a_g = _row_dot(params.W_g, x, u, di) + _row_dot(params.U_g, h_prev, u, dh) + params.b_g[u]
        f, i, o, g = sigmoid(a_f), sigmoid(a_i), sigmoid(a_o), math.tanh(a_g)
        fp, ip, op = sigmoid_prime_from_value(f), sigmoid_prime_from_value(i), sigmoid_prime_from_value(o)
        gp = tanh_prime_from_value(g)
        c_u = f * float(c_prev[u]) + i * g
        tanh_c = math.tanh(c_u)
        dh_dc_u = op * tanh_c                                 # d h[u] / d(c[u]) chained below: o' part + o*tanh'
        o_tanhp = o * (1.0 - tanh_c * tanh_c)                 # o * tanh'(c[u])  -- the cell-state -> hidden path

        # --- derivatives w.r.t. x[j] (only through the gate pre-activations' weight rows W_*[u, j]) ---
        for j in range(di):
            dA_f = params.W_f[u * di + j]
            dA_i = params.W_i[u * di + j]
            dA_o = params.W_o[u * di + j]
            dA_g = params.W_g[u * di + j]
            dc = (float(c_prev[u]) * fp * dA_f) + (g * ip * dA_i) + (i * gp * dA_g)
            dc_dx[u * di + j] = dc
            dh_dx[u * di + j] = tanh_c * op * dA_o + o_tanhp * dc
        # --- derivatives w.r.t. h_prev[k] (through the recurrent weight rows U_*[u, k]) ---
        for k in range(dh):
            dA_f = params.U_f[u * dh + k]
            dA_i = params.U_i[u * dh + k]
            dA_o = params.U_o[u * dh + k]
            dA_g = params.U_g[u * dh + k]
            dc = (float(c_prev[u]) * fp * dA_f) + (g * ip * dA_i) + (i * gp * dA_g)
            dc_dh[u * dh + k] = dc
            dh_dh[u * dh + k] = tanh_c * op * dA_o + o_tanhp * dc
        # --- derivatives w.r.t. c_prev[k]: only the SAME unit (k == u) contributes (c[u] = f*c_prev[u] + ...);
        #     the gates do not see c_prev, so the only path is the f*c_prev[u] term ---
        for k in range(dh):
            if k == u:
                dc = f                                       # d(c[u])/d(c_prev[u]) = f (the forget gate)
                dc_dc[u * dh + k] = dc
                dh_dc[u * dh + k] = o_tanhp * dc             # h[u] = o*tanh(c[u]) -> chain through tanh'
            # k != u: c[u] does not depend on c_prev[k] -> 0 (left as the initialized 0.0)
    return {"dh_dx": dh_dx, "dh_dh_prev": dh_dh, "dh_dc_prev": dh_dc,
            "dc_dx": dc_dx, "dc_dh_prev": dc_dh, "dc_dc_prev": dc_dc}


def lstm_unroll(x_seq: list[float], h0: list[float], c0: list[float],
                params: LstmParams) -> tuple[list[float], list[float]]:
    """Roll the numeric LSTM cell over a sequence (= BPTT over the numeric cells, numerically). ``x_seq`` is
    ``T x input_dim`` flat row-major; ``h0`` / ``c0`` are the length-``hidden_dim`` initial hidden / cell
    states. Returns the FINAL ``(h, c)``. Side-effect-free."""
    di = params.input_dim
    if len(x_seq) % di != 0:
        raise ValueError(f"lstm_unroll: x_seq length {len(x_seq)} not a multiple of input_dim = {di}")
    T = len(x_seq) // di
    h, c = [float(v) for v in h0], [float(v) for v in c0]
    for t in range(T):
        h, c = lstm_cell_reference(x_seq[t * di:(t + 1) * di], h, c, params)
    return h, c


def lstm_states(x_seq: list[float], h0: list[float], c0: list[float],
                params: LstmParams) -> list[tuple[list[float], list[float]]]:
    """Like :func:`lstm_unroll` but returns every ``(h_t, c_t)`` for t in 0..T (length ``T+1``) -- the full
    trajectory, for the temporal-dependence verifier (does ``h_T`` depend on ``x_0``?). Side-effect-free."""
    di = params.input_dim
    T = len(x_seq) // di
    h, c = [float(v) for v in h0], [float(v) for v in c0]
    traj = [(list(h), list(c))]
    for t in range(T):
        h, c = lstm_cell_reference(x_seq[t * di:(t + 1) * di], h, c, params)
        traj.append((list(h), list(c)))
    return traj


def lstm_via_bridge(x_seq: list[float], h0: list[float], c0: list[float], params: LstmParams,
                    group_size: int, bits: int) -> tuple[list[float], list[float]]:
    """E4: the Q8<->float32<->Q8 bridge wrapped around the TRUSTED LSTM unroll (integrate, don't reinvent) --
    the recurrent analog of ``transformer_block_via_bridge`` / ``attention_via_bridge``. The INPUT sequence
    ``x_seq`` arrives as per-group quantized storage; the bridge dequantizes it to float32 and the trusted
    ``lstm_unroll`` runs on the round-tripped value. So the oracle reference for the quantized path is the LSTM
    of the *round-tripped* input -- the SOLE certified error is the INPUT round-trip (R17-certified by
    ``quantization_error_bound``); the gate matmuls/adds are exact and the sigmoid/tanh ride the trusted libm
    edge (treated as exact, like every wrap). Mirrors the transformer/attention bridge exactly."""
    dx = dequantize(quantize_per_group(x_seq, group_size, bits))
    return lstm_unroll(dx, h0, c0, params)


# ============================================================================================================
# Tier B.2 -- the GRU cell: numeric forward reference + the closed-form analytic gradients (the seed).
# ============================================================================================================

@dataclass(frozen=True)
class GruParams:
    """The weights of one GRU cell, all flat row-major. For each gate g in {z (update), r (reset), n
    (candidate)}: ``W_g`` is the input weight ``hidden_dim x input_dim``, ``U_g`` the recurrent weight
    ``hidden_dim x hidden_dim``, ``b_g`` the length-``hidden_dim`` bias. Validated in ``__post_init__``."""

    W_z: list[float]; U_z: list[float]; b_z: list[float]
    W_r: list[float]; U_r: list[float]; b_r: list[float]
    W_n: list[float]; U_n: list[float]; b_n: list[float]
    input_dim: int
    hidden_dim: int

    def __post_init__(self) -> None:
        di, dh = self.input_dim, self.hidden_dim
        if di < 1 or dh < 1:
            raise ValueError(f"gru dims must be >= 1; got input_dim={di} hidden_dim={dh}")
        for nm, W in (("W_z", self.W_z), ("W_r", self.W_r), ("W_n", self.W_n)):
            if len(W) != dh * di:
                raise ValueError(f"gru {nm} must be hidden_dim*input_dim = {dh * di}; got {len(W)}")
        for nm, U in (("U_z", self.U_z), ("U_r", self.U_r), ("U_n", self.U_n)):
            if len(U) != dh * dh:
                raise ValueError(f"gru {nm} must be hidden_dim*hidden_dim = {dh * dh}; got {len(U)}")
        for nm, b in (("b_z", self.b_z), ("b_r", self.b_r), ("b_n", self.b_n)):
            if len(b) != dh:
                raise ValueError(f"gru {nm} must be length hidden_dim = {dh}; got {len(b)}")


def gru_cell_reference(x: list[float], h_prev: list[float], params: GruParams) -> list[float]:
    """ONE standard GRU cell step (numeric, the source of truth). With gate pre-activations:

        z = sigmoid(W_z x + U_z h_prev + b_z)              (update gate)
        r = sigmoid(W_r x + U_r h_prev + b_r)              (reset gate)
        n = tanh(W_n x + r (.) (U_n h_prev) + b_n)         (candidate -- reset applied to the recurrent term)
        h = (1 - z) (.) n + z (.) h_prev                   (the new hidden state)

    This is the canonical PyTorch GRU formulation (the reset gate multiplies the recurrent contribution
    ``U_n h_prev`` BEFORE the bias, not the whole pre-activation). Returns the new ``h`` (fresh length-
    ``hidden_dim`` list). ``sigmoid`` / ``tanh`` are the transcendentals; the rest is exact. Side-effect-free."""
    di, dh = params.input_dim, params.hidden_dim
    if len(x) != di:
        raise ValueError(f"gru_cell: x must be length input_dim = {di}; got {len(x)}")
    if len(h_prev) != dh:
        raise ValueError(f"gru_cell: h_prev must be length hidden_dim = {dh}; got {len(h_prev)}")
    h = [0.0] * dh
    for u in range(dh):
        a_z = _row_dot(params.W_z, x, u, di) + _row_dot(params.U_z, h_prev, u, dh) + params.b_z[u]
        a_r = _row_dot(params.W_r, x, u, di) + _row_dot(params.U_r, h_prev, u, dh) + params.b_r[u]
        z = sigmoid(a_z)
        r = sigmoid(a_r)
        recur_n = _row_dot(params.U_n, h_prev, u, dh)        # U_n h_prev for unit u
        a_n = _row_dot(params.W_n, x, u, di) + r * recur_n + params.b_n[u]
        n = math.tanh(a_n)
        h[u] = (1.0 - z) * n + z * float(h_prev[u])
    return h


def gru_cell_grads(x: list[float], h_prev: list[float], params: GruParams) -> dict:
    """The CLOSED-FORM analytic gradients of one GRU cell's output ``h`` w.r.t. its inputs (``x`` / ``h_prev``),
    via the gate derivatives ``sigma' = s(1-s)`` and ``tanh' = 1 - t^2`` (the seed). Verified against CENTRAL
    finite differences. Returns ``{"dh_dx": hidden_dim x input_dim, "dh_dh_prev": hidden_dim x hidden_dim}``
    (flat row-major Jacobian blocks, ``J[u*? + k]`` = d h[u] / d input[k]).

    The chain per output unit ``u`` (note ``n``'s reset term ``r (.) (U_n h_prev)`` makes ``h[u]`` depend on the
    FULL ``h_prev`` vector through both ``r`` and ``U_n h_prev``):

        z = sigma(a_z),  r = sigma(a_r),  recur = U_n h_prev[u],  a_n = W_n x + r*recur + b_n,  n = tanh(a_n)
        h[u] = (1 - z) n + z h_prev[u]
        d h[u]/d(.) = -z'(a_z) dA_z * n + (1 - z) * n'(a_n) * d a_n
                    + z'(a_z) dA_z * h_prev[u] + z * d(h_prev[u])
        d a_n/d(.) = dA_Wn_x + r'(a_r) dA_r * recur + r * d(recur)

    where ``z' = z(1-z)``, ``r' = r(1-r)``, ``n' = 1 - n^2``; ``dA_z`` / ``dA_r`` are the update/reset gate
    pre-activation derivatives (weight entries), ``dA_Wn_x`` the ``W_n`` row entry (for an ``x`` input; 0 for an
    ``h_prev`` input), ``d(recur) = U_n[u, k]`` for an ``h_prev[k]`` input (0 for an ``x`` input), and
    ``d(h_prev[u])`` is 1 only when differentiating w.r.t. that exact ``h_prev[u]`` (the carry term), else 0.
    Side-effect-free."""
    di, dh = params.input_dim, params.hidden_dim
    if len(x) != di or len(h_prev) != dh:
        raise ValueError("gru_cell_grads: x/h_prev length mismatch with params dims")
    dh_dx = [0.0] * (dh * di)
    dh_dh = [0.0] * (dh * dh)
    for u in range(dh):
        a_z = _row_dot(params.W_z, x, u, di) + _row_dot(params.U_z, h_prev, u, dh) + params.b_z[u]
        a_r = _row_dot(params.W_r, x, u, di) + _row_dot(params.U_r, h_prev, u, dh) + params.b_r[u]
        z = sigmoid(a_z)
        r = sigmoid(a_r)
        recur = _row_dot(params.U_n, h_prev, u, dh)
        a_n = _row_dot(params.W_n, x, u, di) + r * recur + params.b_n[u]
        n = math.tanh(a_n)
        zp = sigmoid_prime_from_value(z)
        rp = sigmoid_prime_from_value(r)
        np_ = tanh_prime_from_value(n)
        hp_u = float(h_prev[u])
        # --- d h[u] / d x[j] ---
        for j in range(di):
            dA_z = params.W_z[u * di + j]
            dA_r = params.W_r[u * di + j]
            dA_Wn = params.W_n[u * di + j]
            d_an = dA_Wn + rp * dA_r * recur                 # d(recur)=0 for an x input
            dh_dx[u * di + j] = (-zp * dA_z * n) + (1.0 - z) * np_ * d_an + (zp * dA_z * hp_u)
        # --- d h[u] / d h_prev[k] ---
        for k in range(dh):
            dA_z = params.U_z[u * dh + k]
            dA_r = params.U_r[u * dh + k]
            d_recur = params.U_n[u * dh + k]                 # d(U_n h_prev[u]) / d h_prev[k]
            d_an = rp * dA_r * recur + r * d_recur           # dA_Wn_x = 0 for an h_prev input
            carry = z if k == u else 0.0                     # z * d(h_prev[u])/d(h_prev[k])
            dh_dh[u * dh + k] = (-zp * dA_z * n) + (1.0 - z) * np_ * d_an + (zp * dA_z * hp_u) + carry
    return {"dh_dx": dh_dx, "dh_dh_prev": dh_dh}


def gru_unroll(x_seq: list[float], h0: list[float], params: GruParams) -> list[float]:
    """Roll the numeric GRU cell over a sequence (= BPTT over the numeric cells). ``x_seq`` is ``T x input_dim``
    flat row-major; ``h0`` is the length-``hidden_dim`` initial state. Returns the FINAL ``h``. Side-effect-free."""
    di = params.input_dim
    if len(x_seq) % di != 0:
        raise ValueError(f"gru_unroll: x_seq length {len(x_seq)} not a multiple of input_dim = {di}")
    T = len(x_seq) // di
    h = [float(v) for v in h0]
    for t in range(T):
        h = gru_cell_reference(x_seq[t * di:(t + 1) * di], h, params)
    return h


def gru_states(x_seq: list[float], h0: list[float], params: GruParams) -> list[list[float]]:
    """Like :func:`gru_unroll` but returns every ``h_t`` for t in 0..T (length ``T+1``) -- the full trajectory,
    for the temporal-dependence verifier. Side-effect-free."""
    di = params.input_dim
    T = len(x_seq) // di
    h = [float(v) for v in h0]
    traj = [list(h)]
    for t in range(T):
        h = gru_cell_reference(x_seq[t * di:(t + 1) * di], h, params)
        traj.append(list(h))
    return traj


def gru_via_bridge(x_seq: list[float], h0: list[float], params: GruParams,
                   group_size: int, bits: int) -> list[float]:
    """E4: the Q8<->float32<->Q8 bridge wrapped around the TRUSTED GRU unroll -- the GRU analog of
    ``lstm_via_bridge``. The INPUT sequence is round-tripped through the bridge then the trusted ``gru_unroll``
    runs; the SOLE certified error is the INPUT round-trip (R17). Side-effect-free."""
    dx = dequantize(quantize_per_group(x_seq, group_size, bits))
    return gru_unroll(dx, h0, params)


# ============================================================================================================
# Independent verifiers (the genuine checks -- recomputed independently of the cell code paths).
# ============================================================================================================

def central_difference_jacobian(forward, base_inputs: list[float], n_out: int,
                                 *, eps: float = 1e-6) -> list[float]:
    """The CENTRAL finite-difference Jacobian of a numeric vector function -- the HARD gate for Tier B's
    closed-form analytic gradients (numeric, since tanh/sigmoid are not closed Tape primitives). ``forward`` is
    a callable ``list[float] -> list[float]`` (length ``n_out``); ``base_inputs`` is the point. Returns a flat
    ``n_out x len(base_inputs)`` row-major Jacobian ``J[o*n_in + k] = (forward(x+eps*e_k)[o] -
    forward(x-eps*e_k)[o]) / (2 eps)`` -- the same central-difference recipe ``autodiff.finite_difference_grad``
    uses, here for a vector output off the Tape."""
    n_in = len(base_inputs)
    jac = [0.0] * (n_out * n_in)
    for k in range(n_in):
        plus = list(base_inputs); plus[k] += eps
        minus = list(base_inputs); minus[k] -= eps
        fp = forward(plus)
        fm = forward(minus)
        for o in range(n_out):
            jac[o * n_in + k] = (fp[o] - fm[o]) / (2.0 * eps)
    return jac


def max_jacobian_error(a: list[float], b: list[float]) -> float:
    """The largest absolute disagreement between two flat Jacobian blocks (for reporting the exact failing entry
    rather than a bare pass/fail), exactly like ``autodiff.max_grad_error``."""
    if len(a) != len(b):
        raise ValueError(f"max_jacobian_error: length mismatch ({len(a)} != {len(b)})")
    return max((abs(ai - bi) for ai, bi in zip(a, b)), default=0.0)


def gate_range_residual(values: list[float], lo: float, hi: float) -> float:
    """How far any gate value strays OUTSIDE the open interval ``(lo, hi)`` -- 0 means every value is strictly
    inside. Used to check every sigmoid output is in ``(0, 1)`` (``gate_range_residual(sigs, 0, 1)``) and every
    tanh output is in ``(-1, 1)`` -- a defining property of the gate nonlinearities (an independent sanity rail
    on the transcendentals). Recomputed directly from the values (not the cell path)."""
    worst = 0.0
    for v in values:
        if v <= lo:
            worst = max(worst, lo - v)
        elif v >= hi:
            worst = max(worst, v - hi)
    return worst


def temporal_dependence(jac_block: list[float], n_out: int, n_in: int) -> float:
    """The MEMORY check: the max absolute entry of a Jacobian block ``d h_T / d x_0`` -- if it is genuinely
    NONZERO the final state depends on the FIRST input, i.e. the cell carries information across time (the whole
    point of a recurrent net). Returns ``max_{u,j} |J[u*n_in + j]|`` -- a value > 0 proves temporal dependence;
    ~0 would mean the early input was forgotten / never propagated."""
    return max((abs(jac_block[u * n_in + j]) for u in range(n_out) for j in range(n_in)), default=0.0)


# ============================================================================================================
# Op-level well-formedness (mirrors check_transformer; NOT a new global R-law).
# ============================================================================================================

_DTYPES = ("f32", "i32")
_CELLS = ("rnn", "lstm", "gru")


def check_recurrent(cell: str, input_dim: int, hidden_dim: int, seq_len: int,
                    in_dtype: str, out_dtype: str) -> list[str]:
    """Op-level well-formedness for a gem.recurrent-cell claim, returned as a list of error strings (empty ==
    well-formed) -- the shape/dtype-law shape gem.matmul / gem.attention / gem.transformer use, lifted into one
    named checker. NOT a globally-numbered R-law (matches ``check_transformer`` / ``check_attention``). The laws:

      1. ``cell`` is a KNOWN recurrent kind ("rnn" / "lstm" / "gru");
      2. extents are POSITIVE: ``input_dim``, ``hidden_dim``, ``seq_len`` all >= 1;
      3. dtype is a KNOWN dtype and PRESERVED: ``out_dtype == in_dtype``;
      4. THE QUARANTINE DTYPE RULE: an LSTM/GRU contains tanh/sigmoid (transcendentals whose libm edges return
         float), so it needs an f32 result -- never i32. (The closed-set relu-RNN is exact and could in principle
         be i32, but the cell family contract here is f32, matching the transformer's quarantine rule -- so the
         rule applies to LSTM/GRU; the relu-RNN is allowed f32 or i32.)"""
    errs: list[str] = []
    if cell not in _CELLS:
        errs.append(f"recurrent: cell {cell!r} is not a known kind {_CELLS}")
    for nm, v in (("input_dim", input_dim), ("hidden_dim", hidden_dim), ("seq_len", seq_len)):
        if v < 1:
            errs.append(f"recurrent: {nm} must be >= 1; got {v}")
    for nm, dt in (("in", in_dtype), ("out", out_dtype)):
        if dt not in _DTYPES:
            errs.append(f"recurrent: {nm} dtype {dt!r} is not a known dtype {_DTYPES}")
    if in_dtype != out_dtype:
        errs.append(f"recurrent: dtype not preserved (in {in_dtype!r} != out {out_dtype!r})")
    # the quarantine dtype rule: LSTM/GRU's tanh/sigmoid (the libm edges return float) need f32.
    if cell in ("lstm", "gru") and in_dtype != "f32":
        errs.append(f"recurrent: {cell} contains tanh/sigmoid (transcendentals; the libm edges return float), "
                    f"so it needs f32, got {in_dtype!r}")
    return errs
