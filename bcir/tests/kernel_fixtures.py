"""PERF-AUDIT's graders: the numeric kernels behind the TMSAO audit groups as they were before
the slice, kept verbatim, and the random inputs the faster kernels are held to them over.

The slice restructured the loops of the telemetry record's schema check, the per-group
quantizer and the quantized group's validation, the int4 packer and unpacker, the squared
distance and the finiteness scan, the reference and tiled matmul, the normal matrix and
right-hand side, the column means and the covariance, the recurrent row dot, the autodiff tape
(the topological order, the forward values, the adjoint accumulation, the gradient), the
training loop's per-batch gradient and the bounded root-PUCT search. A kernel's result must be
BIT-IDENTICAL to the historical one on every input -- every float the same `float.hex`, every
integer and string the same, every refusal the same type and message -- because the audit's
`result_sha256` is a digest of those results and the law of this slice is that no arithmetic
moved: the floating-point operations are the same operations in the same order, only the
Python overhead around them was removed. The row is `kernel.parity`: mismatches over every
input of every kernel, 0 when the slice is exact.
"""

from __future__ import annotations

import math
import random

from bcir.kbcir.autodiff import _BACKWARD, GradResult, Tape
from bcir.kbcir.hardware_rl import (
    _MAX_CANDIDATES,
    _MAX_SIMULATIONS,
    _MAX_U63,
    Q20,
    HardwareSearchResult,
    SearchVisit,
    _int,
    _str,
)
from bcir.kbcir.losses import binary_cross_entropy_with_logits, mse, softmax_cross_entropy
from bcir.kbcir.matmul import _LOOP_ORDERS, TilePlan, _validate_inputs, tile_origins
from bcir.kbcir.quantize import _EXP_MAX, _EXP_MIN, QGroup, _scale_exponent, code_max
from bcir.kbcir.recurrent import sigmoid
from bcir.telemetry import (
    _COUNTER_FIELDS,
    _I64_MAX,
    _MAX_TEXT_FIELD,
    _NORM_FIELDS,
    NORM_MAX,
    NORM_MIN,
)

# --- telemetry.DataDNA.violations, before the slice ----------------------------------------------


def _is_i64(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and -(1 << 63) <= v <= _I64_MAX


def violations_reference(record) -> list[str]:
    bad: list[str] = []
    for f in ("segment_id", "provenance"):
        v = getattr(record, f)
        if not isinstance(v, str) or len(v) > _MAX_TEXT_FIELD or any(ord(ch) < 0x20 for ch in v):
            bad.append(f"{f} not a bounded control-free string")
    if not _is_i64(record.claim_id):
        bad.append("claim_id not a signed-64-bit int")
    elif record.claim_id < 0:
        bad.append(f"claim_id negative ({record.claim_id})")
    for f in _COUNTER_FIELDS:
        v = getattr(record, f)
        if not _is_i64(v):
            bad.append(f"{f} not a signed-64-bit int")
        elif v < 0:
            bad.append(f"{f} negative ({v})")
    for f in _NORM_FIELDS:
        v = getattr(record, f)
        if not _is_i64(v):
            bad.append(f"{f} not a signed-64-bit int")
        elif v < NORM_MIN or v > NORM_MAX:
            bad.append(f"{f} out of [0,100] ({v})")
    return bad


# --- kbcir.quantize, before the slice ------------------------------------------------------------


def _round_half_away(x: float) -> int:
    return int(math.floor(x + 0.5)) if x >= 0.0 else -int(math.floor(-x + 0.5))


def _quantize_code(x: float, cmax: int, scale: float, rounding: str) -> int:
    q = x / scale
    c = int(q) if rounding == "truncate" else _round_half_away(q)  # truncate = toward zero
    return cmax if c > cmax else -cmax if c < -cmax else c  # saturate into the lane


def qgroup_check_reference(codes, scale_exp, bits) -> None:
    """`QGroup.__post_init__` before the slice: the refusals, in their order."""
    cmax = code_max(bits)
    if not isinstance(codes, tuple):
        raise ValueError("QGroup codes must be an immutable tuple")
    if (
        isinstance(scale_exp, bool)
        or not isinstance(scale_exp, int)
        or not _EXP_MIN <= scale_exp <= _EXP_MAX
    ):
        raise ValueError(f"QGroup scale_exp must be an integer in [{_EXP_MIN}, {_EXP_MAX}]")
    for index, code in enumerate(codes):
        if isinstance(code, bool) or not isinstance(code, int) or not -cmax <= code <= cmax:
            raise ValueError(f"QGroup code[{index}] does not fit signed {bits}-bit range")


def quantize_group_reference(values, bits: int, *, rounding: str = "nearest") -> QGroup:
    if rounding not in ("nearest", "truncate"):
        raise ValueError(f"rounding must be 'nearest' or 'truncate'; got {rounding!r}")
    cmax = code_max(bits)
    vals = [float(v) for v in values]
    if any(not math.isfinite(v) for v in vals):
        raise ValueError(
            "quantize_group: inputs must be finite (no inf/nan at the bridge boundary)"
        )
    amax = max((abs(v) for v in vals), default=0.0)
    if amax == 0.0:
        return QGroup(codes=tuple(0 for _ in vals), scale_exp=0, bits=bits)
    e = _scale_exponent(amax, cmax)
    scale = math.ldexp(1.0, e)
    return QGroup(
        codes=tuple(_quantize_code(v, cmax, scale, rounding) for v in vals), scale_exp=e, bits=bits
    )


# --- kbcir.lowbit, before the slice --------------------------------------------------------------


def pack_signed_int4_reference(codes) -> bytes:
    values = tuple(codes)
    if any(type(code) is not int or not -7 <= code <= 7 for code in values):
        raise ValueError("Q4 codes must be integer values in [-7, 7]")
    output = bytearray((len(values) + 1) // 2)
    for index, code in enumerate(values):
        nibble = code & 0xF
        if index & 1:
            output[index // 2] |= nibble << 4
        else:
            output[index // 2] = nibble
    return bytes(output)


def unpack_signed_int4_reference(data, count: int) -> tuple[int, ...]:
    if (
        type(count) is not int
        or count < 0
        or not isinstance(data, (bytes, bytearray, memoryview))
        or len(data) != (count + 1) // 2
    ):
        raise ValueError("packed Q4 byte count does not match the element count")
    raw = bytes(data)
    if count & 1 and raw and raw[-1] & 0xF0:
        raise ValueError("packed Q4 odd-element padding nibble must be zero")
    values = []
    for index in range(count):
        nibble = raw[index // 2] >> (4 if index & 1 else 0) & 0xF
        code = nibble - 16 if nibble & 8 else nibble
        if code == -8:
            raise ValueError("packed Q4 contains forbidden non-symmetric code -8")
        values.append(code)
    return tuple(values)


# --- kbcir.classical / kbcir.unsupervised, before the slice --------------------------------------


def squared_distance_at_reference(a, a_offset: int, b, b_offset: int, length: int) -> float:
    acc = 0.0
    for index in range(length):
        delta = float(a[a_offset + index]) - float(b[b_offset + index])
        acc += delta * delta
    return acc


def kmeans_assign_flat_reference(centroids, point, point_offset: int, k: int, n_feat: int) -> int:
    best_c = 0
    best_d = squared_distance_at_reference(point, point_offset, centroids, 0, n_feat)
    for centroid in range(1, k):
        distance = squared_distance_at_reference(
            point, point_offset, centroids, centroid * n_feat, n_feat
        )
        if distance < best_d:
            best_d = distance
            best_c = centroid
    return best_c


def kmeans_assign_reference(centroids, x, k: int, n_feat: int) -> int:
    if type(k) is not int or k < 1:
        raise ValueError(f"kmeans k must be >= 1; got {k}")
    if type(n_feat) is not int or n_feat < 1:
        raise ValueError(f"kmeans n_feat must be >= 1; got {n_feat}")
    if len(x) != n_feat:
        raise ValueError(f"kmeans point must be length n_feat = {n_feat}; got {len(x)}")
    if len(centroids) != k * n_feat:
        raise ValueError(f"kmeans centroids must be k*n_feat = {k * n_feat}; got {len(centroids)}")
    xf = [float(v) for v in x]
    finite_reference(xf, "kmeans point")
    finite_reference(centroids, "kmeans centroids")
    return kmeans_assign_flat_reference(centroids, xf, 0, k, n_feat)


def finite_reference(values, field: str) -> None:
    try:
        finite = all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} entries must all be finite numbers") from exc
    if not finite:
        raise ValueError(f"{field} entries must all be finite")


# --- kbcir.matmul, before the slice --------------------------------------------------------------


def matmul_reference_reference(a, b, M: int, N: int, K: int) -> list[float]:
    _validate_inputs(a, b, M, N, K)
    c = [0.0] * (M * N)
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += a[i * K + k] * b[k * N + j]
            c[i * N + j] = s
    return c


def matmul_tiled_reference(a, b, M: int, N: int, K: int, plan: "TilePlan") -> list[float]:
    _validate_inputs(a, b, M, N, K)
    if not isinstance(plan, TilePlan):
        raise ValueError("matmul plan must be a TilePlan")
    for name, value in (("tile_m", plan.tile_m), ("tile_n", plan.tile_n), ("tile_k", plan.tile_k)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}")
    if plan.loop_order not in _LOOP_ORDERS:
        raise ValueError(f"unsupported matmul loop order {plan.loop_order!r}")
    c = [0.0] * (M * N)
    tm, tn, tk = plan.tile_m, plan.tile_n, plan.tile_k
    for i0, j0, k0 in tile_origins(M, N, K, plan):
        for i in range(i0, min(i0 + tm, M)):
            for j in range(j0, min(j0 + tn, N)):
                s = 0.0
                for k in range(k0, min(k0 + tk, K)):
                    s += a[i * K + k] * b[k * N + j]
                c[i * N + j] += s
    return c


# --- kbcir.ols / kbcir.pca, before the slice -----------------------------------------------------


def normal_matrix_reference(a: list[float], m: int, n: int) -> list[float]:
    g = [0.0] * (n * n)
    for p in range(n):
        for q in range(n):
            acc = 0.0
            for i in range(m):
                acc += float(a[i * n + p]) * float(a[i * n + q])
            g[p * n + q] = acc
    return g


def normal_rhs_reference(a: list[float], b: list[float], m: int, n: int, nrhs: int) -> list[float]:
    c = [0.0] * (n * nrhs)
    for p in range(n):
        for r in range(nrhs):
            acc = 0.0
            for i in range(m):
                acc += float(a[i * n + p]) * float(b[i * nrhs + r])
            c[p * nrhs + r] = acc
    return c


def column_means_reference(x: list[float], m: int, n: int) -> list[float]:
    means = [0.0] * n
    for i in range(m):
        for j in range(n):
            means[j] += float(x[i * n + j])
    return [mu / m for mu in means]


def center_columns_reference(x: list[float], m: int, n: int) -> list[float]:
    if n < 1:
        raise ValueError(f"pca feature dimension must be >= 1; got n={n}")
    if m < 1:
        raise ValueError(f"pca needs at least one sample m >= 1; got m={m}")
    if len(x) != m * n:
        raise ValueError(f"data matrix must have m*n = {m * n} entries; got {len(x)}")
    means = column_means_reference(x, m, n)
    return [float(x[i * n + j]) - means[j] for i in range(m) for j in range(n)]


def covariance_matrix_reference(x: list[float], m: int, n: int, ddof: int = 1) -> list[float]:
    if ddof < 0:
        raise ValueError(f"ddof must be >= 0; got {ddof}")
    if ddof >= m:
        raise ValueError(
            f"covariance needs ddof < m (a positive divisor m-ddof); got ddof={ddof}, m={m}"
        )
    xc = center_columns_reference(x, m, n)  # validates n>=1, m>=1, len(x)==m*n
    denom = float(m - ddof)
    cmat = [0.0] * (n * n)
    for p in range(n):
        for q in range(n):
            acc = 0.0
            for i in range(m):
                acc += xc[i * n + p] * xc[i * n + q]
            cmat[p * n + q] = acc / denom
    return cmat


# --- kbcir.recurrent, before the slice -----------------------------------------------------------


def row_dot_reference(w: list[float], v: list[float], o: int, d_in: int) -> float:
    acc = 0.0
    for i in range(d_in):
        acc += float(w[o * d_in + i]) * float(v[i])
    return acc


def lstm_cell_reference_reference(x, h_prev, c_prev, params):
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
        a_f = (
            row_dot_reference(params.W_f, x, u, di)
            + row_dot_reference(params.U_f, h_prev, u, dh)
            + params.b_f[u]
        )
        a_i = (
            row_dot_reference(params.W_i, x, u, di)
            + row_dot_reference(params.U_i, h_prev, u, dh)
            + params.b_i[u]
        )
        a_o = (
            row_dot_reference(params.W_o, x, u, di)
            + row_dot_reference(params.U_o, h_prev, u, dh)
            + params.b_o[u]
        )
        a_g = (
            row_dot_reference(params.W_g, x, u, di)
            + row_dot_reference(params.U_g, h_prev, u, dh)
            + params.b_g[u]
        )
        f, i, o, g = sigmoid(a_f), sigmoid(a_i), sigmoid(a_o), math.tanh(a_g)
        c[u] = f * float(c_prev[u]) + i * g
        h[u] = o * math.tanh(c[u])
    return h, c


def gru_cell_reference_reference(x, h_prev, params):
    di, dh = params.input_dim, params.hidden_dim
    if len(x) != di:
        raise ValueError(f"gru_cell: x must be length input_dim = {di}; got {len(x)}")
    if len(h_prev) != dh:
        raise ValueError(f"gru_cell: h_prev must be length hidden_dim = {dh}; got {len(h_prev)}")
    h = [0.0] * dh
    for u in range(dh):
        a_z = (
            row_dot_reference(params.W_z, x, u, di)
            + row_dot_reference(params.U_z, h_prev, u, dh)
            + params.b_z[u]
        )
        a_r = (
            row_dot_reference(params.W_r, x, u, di)
            + row_dot_reference(params.U_r, h_prev, u, dh)
            + params.b_r[u]
        )
        z = sigmoid(a_z)
        r = sigmoid(a_r)
        recur_n = row_dot_reference(params.U_n, h_prev, u, dh)  # U_n h_prev for unit u
        a_n = row_dot_reference(params.W_n, x, u, di) + r * recur_n + params.b_n[u]
        n = math.tanh(a_n)
        h[u] = (1.0 - z) * n + z * float(h_prev[u])
    return h


def lstm_unroll_reference(x_seq, h0, c0, params):
    di = params.input_dim
    if len(x_seq) % di != 0:
        raise ValueError(
            f"lstm_unroll: x_seq length {len(x_seq)} not a multiple of input_dim = {di}"
        )
    T = len(x_seq) // di
    h, c = [float(v) for v in h0], [float(v) for v in c0]
    for t in range(T):
        h, c = lstm_cell_reference_reference(x_seq[t * di : (t + 1) * di], h, c, params)
    return h, c


def gru_unroll_reference(x_seq, h0, params):
    di = params.input_dim
    if len(x_seq) % di != 0:
        raise ValueError(
            f"gru_unroll: x_seq length {len(x_seq)} not a multiple of input_dim = {di}"
        )
    T = len(x_seq) // di
    h = [float(v) for v in h0]
    for t in range(T):
        h = gru_cell_reference_reference(x_seq[t * di : (t + 1) * di], h, params)
    return h


# --- kbcir.autodiff, before the slice ------------------------------------------------------------


def topo_order_reference(tape: Tape, root: int) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    stack = [(root, False)]
    while stack:
        nid, expanded = stack.pop()
        if expanded:
            if nid not in seen:
                seen.add(nid)
                order.append(nid)
            continue
        if nid in seen:
            continue
        stack.append((nid, True))
        for a in tape.node(nid).args:
            if a not in seen:
                stack.append((a, False))
    return order


def _forward_values_reference(tape: Tape, topo: list[int], env: dict) -> dict:
    cache: dict[int, float] = {}
    for nid in topo:
        n = tape.node(nid)
        op = n.op
        if op == "const":
            cache[nid] = n.const
        elif op == "var":
            if n.const not in env:
                raise KeyError(f"unbound var {n.const!r}")
            cache[nid] = float(env[n.const])
        elif op == "neg":
            cache[nid] = -cache[n.args[0]]
        elif op == "add":
            cache[nid] = cache[n.args[0]] + cache[n.args[1]]
        elif op == "sub":
            cache[nid] = cache[n.args[0]] - cache[n.args[1]]
        elif op == "mul":
            cache[nid] = cache[n.args[0]] * cache[n.args[1]]
        elif op == "div":
            cache[nid] = cache[n.args[0]] / cache[n.args[1]]
        elif op == "exp":
            cache[nid] = math.exp(cache[n.args[0]])
        elif op == "log":
            cache[nid] = math.log(cache[n.args[0]])
        elif op == "sqrt":
            cache[nid] = math.sqrt(cache[n.args[0]])
        elif op == "tanh":
            cache[nid] = math.tanh(cache[n.args[0]])
        elif op == "sin":
            cache[nid] = math.sin(cache[n.args[0]])
        elif op == "cos":
            cache[nid] = math.cos(cache[n.args[0]])
        elif op == "select":
            cond, a, b = n.args
            cache[nid] = cache[a] if cache[cond] > 0 else cache[b]
        elif op == "dot":
            k = n.const[1]
            us, vs = n.args[:k], n.args[k:]
            cache[nid] = sum(cache[u] * cache[v] for u, v in zip(us, vs))
        else:  # pragma: no cover
            raise ValueError(f"unknown op {op!r}")
    return cache


def evaluate_reference(tape: Tape, root: int, env: dict) -> float:
    return _forward_values_reference(tape, topo_order_reference(tape, root), env)[root]


def _bw_dot_reference(fvals, args, const, gz):
    k = const[1]
    us, vs = args[:k], args[k:]
    out = []
    for u, v in zip(us, vs):
        out.append((u, gz * fvals[v]))
        out.append((v, gz * fvals[u]))
    return out


_BACKWARD_REFERENCE = {**_BACKWARD, "dot": _bw_dot_reference}


def _accumulate_adjoints_reference(tape: Tape, root: int, order, fvals: dict) -> tuple[dict, int]:
    adj: dict[int, float] = {root: 1.0}  # seed: d(output)/d(output) = 1
    firings = 0
    for nid in order:
        gz = adj.get(nid, 0.0)
        if gz == 0.0:
            continue  # no adjoint flows here -> rule is a no-op
        n = tape.node(nid)
        rule = _BACKWARD_REFERENCE.get(n.op)
        if rule is None:
            continue  # a leaf (const/var): differentiation boundary
        firings += 1
        for operand, contribution in rule(fvals, n.args, n.const, gz):
            adj[operand] = adj.get(operand, 0.0) + contribution
    return adj, firings


def grad_reference(
    tape: Tape, output: int, inputs, *, order: list[int] | None = None
) -> GradResult:
    if not isinstance(inputs, dict):
        raise TypeError("grad() needs an env dict (var name -> value); use grad_at for names + env")
    env = inputs
    topo = topo_order_reference(tape, output)
    fvals = _forward_values_reference(tape, topo, env)
    rev = order if order is not None else list(reversed(topo))
    adj, firings = _accumulate_adjoints_reference(tape, output, rev, fvals)
    name_to_nid = {tape.node(nid).const: nid for nid in topo if tape.node(nid).op == "var"}
    grads = {name: adj.get(name_to_nid.get(name, -1), 0.0) for name in env}
    forward_ops = sum(1 for nid in topo if tape.node(nid).op in _BACKWARD_REFERENCE)
    return GradResult(
        grads=grads, value=fvals[output], forward_ops=forward_ops, backward_ops=firings
    )


# --- kbcir.training._batch_loss_and_grad, before the slice ---------------------------------------


def batch_loss_and_grad_reference(model, params, param_names, xb, yb, loss):
    env = dict(zip(param_names, params))
    tape = Tape()
    out = model(tape, param_names, xb)  # the model builds the forward into `tape`; returns node(s)
    nb = len(xb)

    if loss == "mse":
        pred_nids = tuple(out)
        loss_node = mse(tape, pred_nids, list(yb))
        loss_val = evaluate_reference(tape, loss_node, env)
        g = grad_reference(tape, loss_node, env).grads
        return loss_val, {name: g.get(name, 0.0) for name in param_names}

    grad_acc = {name: 0.0 for name in param_names}
    total_loss = 0.0
    if loss == "bce":
        for logit_node, y in zip(out, yb):
            z = evaluate_reference(tape, logit_node, env)
            lval, grad_logits = binary_cross_entropy_with_logits(
                [z], [y]
            )  # mean over 1 -> the per-example loss
            seed = grad_logits[0]  # dL/dz = sigmoid(z) - y
            dlogit = grad_reference(tape, logit_node, env).grads  # d(logit)/dparam, seed 1.0
            for name in param_names:
                grad_acc[name] += seed * dlogit.get(name, 0.0)
            total_loss += lval
        loss_val = total_loss / nb
        return loss_val, {name: grad_acc[name] / nb for name in param_names}

    for logit_row, y in zip(out, yb):
        zvec = [evaluate_reference(tape, ln, env) for ln in logit_row]
        K = len(zvec)
        onehot = [1.0 if k == int(round(y)) else 0.0 for k in range(K)]
        lval, grad_logits = softmax_cross_entropy(zvec, onehot)  # grad = softmax(z) - onehot
        for k, ln in enumerate(logit_row):
            seed = grad_logits[k]
            dlogit = grad_reference(tape, ln, env).grads  # d(logit_k)/dparam
            for name in param_names:
                grad_acc[name] += seed * dlogit.get(name, 0.0)
        total_loss += lval
    loss_val = total_loss / nb
    return loss_val, {name: grad_acc[name] / nb for name in param_names}


def predict_reference(model, params, param_names, X, loss):
    env = dict(zip(param_names, params))
    tape = Tape()
    out = model(tape, param_names, X)
    if loss == "mse":
        return [evaluate_reference(tape, nid, env) for nid in out]
    if loss == "bce":
        return [1.0 / (1.0 + math.exp(-evaluate_reference(tape, nid, env))) for nid in out]
    preds = []
    for logit_row in out:
        z = [evaluate_reference(tape, ln, env) for ln in logit_row]
        m = max(z)
        ex = [math.exp(v - m) for v in z]
        s = sum(ex) or 1.0
        preds.append([e / s for e in ex])
    return preds


# --- kbcir.hardware_rl.bounded_mcts, before the slice --------------------------------------------


def bounded_mcts_reference(
    candidate_ids, priors_q20, evaluator, *, simulations: int = 64, exploration_q20: int = 2 * Q20
) -> HardwareSearchResult:
    try:
        candidates = tuple(candidate_ids)
    except TypeError as exc:
        raise ValueError("MCTS candidate IDs must be a bounded sequence") from exc
    if (
        not candidates
        or len(candidates) > _MAX_CANDIDATES
        or any(not isinstance(row, str) or not row for row in candidates)
    ):
        raise ValueError("MCTS needs 1..256 unique candidate IDs")
    for candidate in candidates:
        _str(candidate, "MCTS candidate ID")
    if len(set(candidates)) != len(candidates):
        raise ValueError("MCTS needs 1..256 unique candidate IDs")
    candidates = tuple(sorted(candidates))
    _int(simulations, "MCTS simulations", minimum=len(candidates), maximum=_MAX_SIMULATIONS)
    _int(exploration_q20, "MCTS exploration_q20", maximum=100 * Q20)
    if (
        not callable(evaluator)
        or not isinstance(priors_q20, dict)
        or set(priors_q20) != set(candidates)
    ):
        raise ValueError("MCTS requires an evaluator and one prior per candidate")
    if (
        any(type(value) is not int or not 0 <= value <= _MAX_U63 for value in priors_q20.values())
        or sum(priors_q20.values()) <= 0
    ):
        raise ValueError("MCTS priors must be non-negative integers with positive mass")
    total_prior = sum(priors_q20.values())
    visits = {candidate: 0 for candidate in candidates}
    utility = {candidate: 0 for candidate in candidates}
    for step in range(simulations):
        if step < len(candidates):
            selected = candidates[step]
        else:
            root = math.isqrt(step + 1)

            def puct(candidate, root=root):
                mean = utility[candidate] // visits[candidate]
                explore = (
                    exploration_q20
                    * priors_q20[candidate]
                    * root
                    // (total_prior * (visits[candidate] + 1))
                )
                return mean + explore

            selected = min(candidates, key=lambda candidate: (-puct(candidate), candidate))
        value = evaluator(selected)
        if type(value) is not int or not -_MAX_U63 <= value <= _MAX_U63:
            raise ValueError("MCTS evaluator must return a bounded integer utility")
        visits[selected] += 1
        utility[selected] += value
    selected = min(
        candidates,
        key=lambda candidate: (
            -visits[candidate],
            -(utility[candidate] // visits[candidate]),
            candidate,
        ),
    )
    rows = tuple(
        SearchVisit(candidate, visits[candidate], utility[candidate] // visits[candidate])
        for candidate in candidates
    )
    return HardwareSearchResult(selected, simulations, rows)


# --- the grader --------------------------------------------------------------------------------------


def bits(value):
    """`value` as an exactly comparable structure: every float its `hex`, containers walked."""
    if isinstance(value, float):
        return ("f", value.hex())
    if isinstance(value, bool):
        return ("b", value)
    if isinstance(value, int):
        return ("i", value)
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(bits(v) for v in value))
    if isinstance(value, dict):
        return ("d", tuple((bits(k), bits(v)) for k, v in value.items()))
    if hasattr(value, "__dataclass_fields__"):
        return (
            type(value).__name__,
            tuple((f, bits(getattr(value, f))) for f in value.__dataclass_fields__),
        )
    return ("o", repr(value))


def outcome(fn, *args, **kwargs) -> tuple:
    """`fn(...)` as a comparable value: its result in `bits`, or the refusal's type and message."""
    try:
        return ("ok", bits(fn(*args, **kwargs)))
    except Exception as exc:  # noqa: BLE001 -- the grader compares every outcome, crashes included
        return ("raise", type(exc).__name__, str(exc))


_ODD_FLOATS = (0.0, -0.0, 1.0, -1.0, 0.5, 1e-300, 5e-324, 1e300, 123456789.125, 3.0, 1 << 60)


def _floats(rng: random.Random, count: int, *, odd: bool = True) -> list:
    """`count` values: floats of every magnitude, the odd integer and bool, infinities and NaN
    when `odd` (the refusals' inputs)."""
    out = []
    for _ in range(count):
        pick = rng.random()
        if pick < 0.6:
            out.append(rng.uniform(-10.0, 10.0))
        elif pick < 0.75:
            out.append(rng.choice(_ODD_FLOATS))
        elif pick < 0.85:
            out.append(rng.randint(-5, 5))
        elif odd and pick < 0.9:
            out.append(rng.choice((math.inf, -math.inf, math.nan)))
        elif odd and pick < 0.93:
            out.append(rng.choice((True, "7", None)))
        else:
            out.append(rng.uniform(-1e6, 1e6))
    return out


class _Record:
    """A record with the DataDNA fields, any values (the schema check reads attributes)."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def _telemetry_cases(rng: random.Random) -> list:
    from bcir.telemetry import DataDNA

    cases = []
    for _ in range(300):
        text = rng.choice(
            (
                "tmsao",
                "",
                "a" * _MAX_TEXT_FIELD,
                "a" * (_MAX_TEXT_FIELD + 1),
                "x\x01y",
                "\x1f",
                " ok ",
                "é",
                7,
                None,
                b"b",
            )
        )
        ints = []
        for _ in range(7):
            pick = rng.random()
            if pick < 0.5:
                ints.append(rng.randint(0, 100))
            elif pick < 0.65:
                ints.append(rng.randint(101, 10**6))
            elif pick < 0.75:
                ints.append(rng.randint(-(10**6), -1))
            elif pick < 0.82:
                ints.append(rng.choice((_I64_MAX, _I64_MAX + 1, -(1 << 63), -(1 << 63) - 1)))
            elif pick < 0.9:
                ints.append(rng.choice((True, False, 1.0, 0.5, math.nan, None, "3")))
            else:
                ints.append(rng.randint(0, 1 << 40))
        fields = dict(
            segment_id=text if rng.random() < 0.5 else rng.choice(("seg", "s\x00", 3)),
            claim_id=ints[0],
            cycles=ints[1],
            bytes=ints[2],
            misses=ints[3],
            thermal=ints[4],
            voltage=ints[5],
            utilization=ints[6],
            provenance=text,
        )
        record = DataDNA(**fields) if rng.random() < 0.5 else _Record(**fields)
        cases.append(record)
    return cases


def _random_plan(rng: random.Random, M: int, N: int, K: int) -> TilePlan:
    from bcir.kbcir.matmul import plan_matmul

    if rng.random() < 0.3:
        return plan_matmul(M, N, K)
    return TilePlan(
        rng.randint(1, M + 1),
        rng.randint(1, N + 1),
        rng.randint(1, K + 1),
        rng.choice(_LOOP_ORDERS),
        1,
        1,
        1,
        True,
    )


def _random_tape(rng: random.Random):
    """A random DAG over a few named inputs and constants, every op reachable; the env and
    its root. Domains are kept where the transcendental ops are defined, or not -- a refusal
    is an outcome too."""
    tape = Tape()
    names = [f"w{i}" for i in range(rng.randint(1, 4))]
    env = {name: rng.uniform(0.1, 3.0) for name in names}
    if rng.random() < 0.3:
        env["unused"] = 1.0
    nodes = [tape.var(name) for name in names] + [
        tape.const(rng.choice((0.5, 2.0, -1.5, 3.0))) for _ in range(2)
    ]
    for _ in range(rng.randint(1, 14)):
        op = rng.choice(
            (
                "add",
                "sub",
                "mul",
                "neg",
                "div",
                "exp",
                "log",
                "sqrt",
                "tanh",
                "sin",
                "cos",
                "select",
                "dot",
                "dup",
            )
        )
        a, b = rng.choice(nodes), rng.choice(nodes)
        if op == "dup":
            nodes.append(tape.mul(a, a))  # a shared operand: one node, two uses
        elif op in ("add", "sub", "mul", "div"):
            nodes.append(getattr(tape, op)(a, b))
        elif op == "select":
            nodes.append(tape.select(rng.choice(nodes), a, b))
        elif op == "dot":
            k = rng.randint(1, 3)
            nodes.append(
                tape.dot(
                    tuple(rng.choice(nodes) for _ in range(k)),
                    tuple(rng.choice(nodes) for _ in range(k)),
                )
            )
        else:
            nodes.append(getattr(tape, op)(a))
    return tape, env, nodes[-1]


def _mcts_cases(rng: random.Random) -> list:
    cases = []
    for _ in range(120):
        count = rng.randint(1, 9)
        candidates = [f"plan-{rng.randint(0, 40):02d}" for _ in range(count)]
        if rng.random() < 0.1:
            candidates.append(candidates[0])  # a duplicate: refused
        priors = {c: rng.randint(0, 3 * Q20) for c in candidates}
        if rng.random() < 0.05:
            priors[candidates[0]] = -1
        utility = {c: rng.randint(-3 * Q20, 3 * Q20) for c in candidates}
        simulations = rng.randint(1, 80)
        exploration = rng.choice((Q20, 2 * Q20, 0, 50 * Q20))
        cases.append((tuple(candidates), priors, utility.__getitem__, simulations, exploration))
    return cases


def _qgroup_check(codes, scale_exp, bits) -> None:
    """The production validation as the reference spells it: None, or the refusal."""
    QGroup(codes, scale_exp, bits)


def rails(seed: int) -> list[tuple[str, tuple, tuple]]:
    """Every (rail, reference outcome, outcome) of one seed's inputs."""
    from bcir.kbcir.autodiff import _topo_order, evaluate, grad
    from bcir.kbcir.classical import _squared_distance_at
    from bcir.kbcir.hardware_rl import bounded_mcts
    from bcir.kbcir.lowbit import pack_signed_int4, unpack_signed_int4
    from bcir.kbcir.matmul import matmul_reference, matmul_tiled
    from bcir.kbcir.ols import _normal_matrix, _normal_rhs
    from bcir.kbcir.pca import _column_means, center_columns, covariance_matrix
    from bcir.kbcir.quantize import quantize_group
    from bcir.kbcir.recurrent import (
        GruParams,
        LstmParams,
        _row_dot,
        gru_cell_reference,
        gru_unroll,
        lstm_cell_reference,
        lstm_unroll,
    )
    from bcir.kbcir.training import (
        _batch_loss_and_grad,
        _predict,
        linear_model,
        mlp_model,
        mlp_param_names,
    )
    from bcir.kbcir.unsupervised import _finite, _kmeans_assign_flat, kmeans_assign
    from bcir.telemetry import DataDNA

    rng = random.Random(seed)
    out: list[tuple[str, tuple, tuple]] = []
    # the telemetry schema
    for record in _telemetry_cases(rng):
        out.append(
            (
                "violations",
                outcome(violations_reference, record),
                outcome(DataDNA.violations, record),
            )
        )
    # the quantizer and the group's validation
    for _ in range(40):
        values = _floats(rng, rng.randint(0, 40))
        bits_ = rng.choice((2, 3, 4, 8, 16, 1, 4097))
        rounding = rng.choice(("nearest", "nearest", "truncate", "banker"))
        out.append(
            (
                f"quantize_group[{bits_}]",
                outcome(quantize_group_reference, values, bits_, rounding=rounding),
                outcome(quantize_group, values, bits_, rounding=rounding),
            )
        )
    for _ in range(40):
        cmax = code_max(rng.choice((2, 4, 8)))
        codes = [
            rng.randint(-cmax - 2, cmax + 2) if rng.random() < 0.9 else rng.choice((True, 1.0, "1"))
            for _ in range(rng.randint(0, 12))
        ]
        codes = tuple(codes) if rng.random() < 0.9 else codes
        scale_exp = rng.choice((0, -3, 7, _EXP_MIN, _EXP_MAX, _EXP_MAX + 1, True, 1.0))
        bits_ = 2 if cmax == 1 else (4 if cmax == 7 else 8)
        out.append(
            (
                "qgroup",
                outcome(qgroup_check_reference, codes, scale_exp, bits_),
                outcome(_qgroup_check, codes, scale_exp, bits_),
            )
        )
    # the int4 packer and unpacker
    for _ in range(40):
        codes = [
            rng.randint(-8, 8) if rng.random() < 0.95 else rng.choice((True, 1.0))
            for _ in range(rng.randint(0, 20))
        ]
        out.append(
            (
                "pack_int4",
                outcome(pack_signed_int4_reference, codes),
                outcome(pack_signed_int4, codes),
            )
        )
        count = rng.randint(0, 20)
        data = bytes(
            rng.randint(0, 255) for _ in range((count + 1) // 2 + rng.choice((0, 0, 0, 1)))
        )
        if rng.random() < 0.5 and count & 1 and data:
            data = data[:-1] + bytes([data[-1] & 0x0F])  # the padding nibble zeroed
        out.append(
            (
                "unpack_int4",
                outcome(unpack_signed_int4_reference, data, count),
                outcome(unpack_signed_int4, data, count),
            )
        )
    # the squared distance and the finiteness scan
    for _ in range(40):
        length = rng.randint(0, 12)
        a = _floats(rng, length + 3, odd=rng.random() < 0.2)
        b = _floats(rng, length + 3, odd=rng.random() < 0.2)
        ao, bo = rng.randint(0, 3), rng.randint(0, 3)
        out.append(
            (
                "squared_distance",
                outcome(squared_distance_at_reference, a, ao, b, bo, length),
                outcome(_squared_distance_at, a, ao, b, bo, length),
            )
        )
        out.append(("finite", outcome(finite_reference, a, "x"), outcome(_finite, a, "x")))
    # the nearest centroid (float storage, as its callers hand it) and the public assign
    for _ in range(30):
        k, n_feat = rng.randint(1, 5), rng.randint(1, 6)
        centroids = [float(v) for v in _floats(rng, k * n_feat, odd=False)]
        points = [float(v) for v in _floats(rng, 3 * n_feat, odd=False)]
        if rng.random() < 0.3:  # a tie: two centroids alike
            centroids[:n_feat] = centroids[-n_feat:]
        offset = rng.randint(0, 2) * n_feat
        out.append(
            (
                "kmeans_assign_flat",
                outcome(kmeans_assign_flat_reference, centroids, points, offset, k, n_feat),
                outcome(_kmeans_assign_flat, centroids, points, offset, k, n_feat),
            )
        )
        raw = _floats(rng, k * n_feat, odd=rng.random() < 0.2)
        x = _floats(rng, n_feat + rng.choice((0, 0, 1)), odd=rng.random() < 0.2)
        out.append(
            (
                "kmeans_assign",
                outcome(kmeans_assign_reference, raw, x, k, n_feat),
                outcome(kmeans_assign, raw, x, k, n_feat),
            )
        )
    # the matmuls
    for _ in range(12):
        M, N, K = rng.randint(1, 7), rng.randint(1, 7), rng.randint(1, 7)
        a = _floats(rng, M * K, odd=False)
        b = _floats(rng, K * N, odd=False)
        plan = _random_plan(rng, M, N, K)
        out.append(
            (
                "matmul_reference",
                outcome(matmul_reference_reference, a, b, M, N, K),
                outcome(matmul_reference, a, b, M, N, K),
            )
        )
        out.append(
            (
                "matmul_tiled",
                outcome(matmul_tiled_reference, a, b, M, N, K, plan),
                outcome(matmul_tiled, a, b, M, N, K, plan),
            )
        )
    out.append(
        (
            "matmul_short",
            outcome(matmul_reference_reference, [1.0], [1.0, 2.0], 1, 1, 1),
            outcome(matmul_reference, [1.0], [1.0, 2.0], 1, 1, 1),
        )
    )
    # the normal equations and the covariance
    for _ in range(12):
        m, n, nrhs = rng.randint(1, 9), rng.randint(1, 5), rng.randint(1, 3)
        a = _floats(rng, m * n, odd=False)
        b = _floats(rng, m * nrhs, odd=False)
        ddof = rng.choice((0, 1, 1, m))
        out.append(
            (
                "normal_matrix",
                outcome(normal_matrix_reference, a, m, n),
                outcome(_normal_matrix, a, m, n),
            )
        )
        out.append(
            (
                "normal_rhs",
                outcome(normal_rhs_reference, a, b, m, n, nrhs),
                outcome(_normal_rhs, a, b, m, n, nrhs),
            )
        )
        out.append(
            (
                "column_means",
                outcome(column_means_reference, a, m, n),
                outcome(_column_means, a, m, n),
            )
        )
        out.append(
            (
                "center_columns",
                outcome(center_columns_reference, a, m, n),
                outcome(center_columns, a, m, n),
            )
        )
        out.append(
            (
                "covariance",
                outcome(covariance_matrix_reference, a, m, n, ddof),
                outcome(covariance_matrix, a, m, n, ddof),
            )
        )
    # the recurrent row dot
    for _ in range(20):
        d_in, d_out = rng.randint(1, 9), rng.randint(1, 4)
        w = _floats(rng, d_in * d_out, odd=False)
        v = _floats(rng, d_in + rng.randint(-1, 2), odd=False)  # one short: refused alike
        o = rng.randint(0, d_out - 1)
        out.append(
            ("row_dot", outcome(row_dot_reference, w, v, o, d_in), outcome(_row_dot, w, v, o, d_in))
        )
    # the recurrent cells and their unrolls (the weights as floats or ints, a wrong length refused)
    for _ in range(12):
        di, dh = rng.randint(1, 4), rng.randint(1, 4)
        steps = rng.randint(0, 4)
        vec = lambda count: [
            rng.uniform(-1.5, 1.5) if rng.random() < 0.85 else rng.randint(-2, 2)
            for _ in range(count)
        ]
        lstm = LstmParams(
            vec(dh * di),
            vec(dh * dh),
            vec(dh),
            vec(dh * di),
            vec(dh * dh),
            vec(dh),
            vec(dh * di),
            vec(dh * dh),
            vec(dh),
            vec(dh * di),
            vec(dh * dh),
            vec(dh),
            di,
            dh,
        )
        gru = GruParams(
            vec(dh * di),
            vec(dh * dh),
            vec(dh),
            vec(dh * di),
            vec(dh * dh),
            vec(dh),
            vec(dh * di),
            vec(dh * dh),
            vec(dh),
            di,
            dh,
        )
        x = vec(di + rng.choice((0, 0, 0, 1)))
        h0 = vec(dh + rng.choice((0, 0, 0, 1)))
        c0 = vec(dh + rng.choice((0, 0, 0, 1)))
        seq = vec(steps * di + rng.choice((0, 0, 0, 1)))
        out.append(
            (
                "lstm_cell",
                outcome(lstm_cell_reference_reference, x, h0, c0, lstm),
                outcome(lstm_cell_reference, x, h0, c0, lstm),
            )
        )
        out.append(
            (
                "gru_cell",
                outcome(gru_cell_reference_reference, x, h0, gru),
                outcome(gru_cell_reference, x, h0, gru),
            )
        )
        out.append(
            (
                "lstm_unroll",
                outcome(lstm_unroll_reference, seq, h0, c0, lstm),
                outcome(lstm_unroll, seq, h0, c0, lstm),
            )
        )
        out.append(
            (
                "gru_unroll",
                outcome(gru_unroll_reference, seq, h0, gru),
                outcome(gru_unroll, seq, h0, gru),
            )
        )
    # the autodiff tape
    for _ in range(25):
        tape, env, root = _random_tape(rng)
        out.append(
            (
                "topo_order",
                outcome(topo_order_reference, tape, root),
                outcome(_topo_order, tape, root),
            )
        )
        out.append(
            (
                "evaluate",
                outcome(evaluate_reference, tape, root, env),
                outcome(evaluate, tape, root, env),
            )
        )
        out.append(
            ("grad", outcome(grad_reference, tape, root, env), outcome(grad, tape, root, env))
        )
        if rng.random() < 0.3:
            missing = {k: v for k, v in env.items() if k != "w0"}
            out.append(
                (
                    "grad[unbound]",
                    outcome(grad_reference, tape, root, missing),
                    outcome(grad, tape, root, missing),
                )
            )
    # the training loop's per-batch gradient
    for _ in range(10):
        d = rng.randint(1, 3)
        nb = rng.randint(1, 5)
        loss = rng.choice(("bce", "mse", "softmax_ce"))
        if loss == "softmax_ce":
            k = rng.randint(2, 3)
            names = [f"w{i}" for i in range(d * k + k)]
            params = [rng.uniform(-1, 1) for _ in names]

            def model(tape, param_names, X, d=d, k=k):
                rows = []
                for row in X:
                    logits = []
                    for c in range(k):
                        ws = tuple(tape.var(param_names[c * d + i]) for i in range(d))
                        xs = tuple(tape.const(row[i]) for i in range(d))
                        logits.append(tape.add(tape.dot(ws, xs), tape.var(param_names[d * k + c])))
                    rows.append(logits)
                return rows

            yb = tuple(float(rng.randint(0, k - 1)) for _ in range(nb))
        elif rng.random() < 0.5:
            names = mlp_param_names(d, 2)
            params = [rng.uniform(-1, 1) for _ in names]
            model = mlp_model(d, 2)
            yb = tuple(
                float(rng.randint(0, 1)) if loss == "bce" else rng.uniform(-2, 2) for _ in range(nb)
            )
        else:
            names = [f"w{i}" for i in range(d)] + ["b"]
            params = [rng.uniform(-1, 1) for _ in names]
            model = linear_model
            yb = tuple(
                float(rng.randint(0, 1)) if loss == "bce" else rng.uniform(-2, 2) for _ in range(nb)
            )
        xb = tuple(tuple(rng.uniform(-3, 3) for _ in range(d)) for _ in range(nb))
        out.append(
            (
                f"predict[{loss}]",
                outcome(predict_reference, model, params, names, xb, loss),
                outcome(_predict, model, params, names, xb, loss),
            )
        )
        out.append(
            (
                f"batch[{loss}]",
                outcome(batch_loss_and_grad_reference, model, params, names, xb, yb, loss),
                outcome(_batch_loss_and_grad, model, params, names, xb, yb, loss),
            )
        )
    # the bounded search
    for candidates, priors, evaluator, simulations, exploration in _mcts_cases(rng):
        out.append(
            (
                "mcts",
                outcome(
                    bounded_mcts_reference,
                    candidates,
                    priors,
                    evaluator,
                    simulations=simulations,
                    exploration_q20=exploration,
                ),
                outcome(
                    bounded_mcts,
                    candidates,
                    priors,
                    evaluator,
                    simulations=simulations,
                    exploration_q20=exploration,
                ),
            )
        )
    return out


def measure(seen: dict | None = None, seeds: int = 24) -> dict[str, float]:
    """The row (module docstring), over `seeds` seeds of inputs and every kernel."""
    mismatches = 0
    stats: dict = {"items": 0, "refused": 0, "rails": {}, "messages": set()}
    for seed in range(seeds):
        for rail, ref, new in rails(seed):
            stats["items"] += 1
            stats["rails"][rail] = stats["rails"].get(rail, 0) + 1
            if ref[0] == "raise":
                stats["refused"] += 1
                stats["messages"].add(ref[2][:50])
            if new != ref:
                mismatches += 1
                if seen is not None:
                    seen.setdefault("mismatched", []).append((seed, rail, ref, new))
    if seen is not None:
        seen.update(stats)
    return {"kernel.parity": float(mismatches)}


__all__ = ["bits", "measure", "outcome", "rails"]
