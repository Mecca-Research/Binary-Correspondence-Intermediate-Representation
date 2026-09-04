"""G6 (roadmap task #53): lower the B3 autodiff oracle to an emitted forward/backward C kernel + SGD step.

The VISION_ALIGNMENT_AUDIT (Pillar 5b) flags that BCIR's autodiff (``bcir.kbcir.autodiff``, the B3
oracle) is a complete, correct, content-addressed reverse-mode autodiff organ -- forward eval, reverse +
symbolic gradients, Hessians, finite-difference-gated -- but is **Python-oracle-only, kept cold, NOT
lowered**, so gradients/training cannot run on the deployed bare-metal rail. This module is that lowering:
an emitter that takes an autodiff computation graph (a ``Tape`` + a designated output node + the ordered
parameter/input set) and emits a self-contained C kernel that computes the **forward value AND the
reverse-mode backward-pass gradients**, plus a **minimal SGD weight-update step** -- the deployable
training primitive.

HONEST DEPTH (the precise boundary). This is the *deployable training primitive* -- a forward + backward +
one-SGD-step C kernel, reference-verified against the oracle's reverse-mode ``grad`` AND its
finite-difference gradient (the B3 hard gate) to float round-off. It is NOT a full optimizer / data-loader
/ mini-batch / momentum / autodiff-of-control-flow framework: that breadth is out of scope. What is
delivered is the verifiable gradient+update kernel -- "a significant portion of training on bare-metal C"
in the smallest honest, parity-gated form. A short training loop (forward -> backward -> SGD) is shown to
strictly DECREASE a simple loss, in both the oracle and the compiled C, agreeing.

HOW THE BACKWARD PASS IS LOWERED. The forward DAG is a FIXED dag once the ``Tape`` is built, so the
emitter BAKES the topological order (exactly ``autodiff._topo_order``) into straight-line C -- no graph
walk at runtime, mirroring the inference.py baking discipline. Each node ``z`` gets a forward temporary
``float v<id> = <op of its operands>;`` (the C translation of ``autodiff.evaluate`` / ``_forward_values``).
The reverse pass then declares an adjoint accumulator ``float g<id> = 0.0f;`` per node, seeds the OUTPUT
adjoint ``g<root> = 1.0f`` (d(out)/d(out) = 1), and walks nodes in REVERSE topo order emitting the C
translation of each primitive's local ``_BACKWARD`` rule as ``+=`` accumulations into its operands'
adjoints:

    neg:    g_a += -g_z
    add:    g_a += g_z;            g_b += g_z
    sub:    g_a += g_z;            g_b += -g_z
    mul:    g_a += g_z * v_b;      g_b += g_z * v_a
    div:    g_a += g_z / v_b;      g_b += -g_z * v_a / (v_b*v_b)
    dot:    for each i: g_{u_i} += g_z * v_{v_i};  g_{v_i} += g_z * v_{u_i}
    select: (cond>0)  g_a += g_z, g_b += 0, g_cond += 0   else  g_a += 0, g_b += g_z, g_cond += 0

These are EXACTLY ``autodiff._BACKWARD`` (``_bw_neg`` / ``_bw_add`` / ... / ``_bw_select``) transcribed to
C. Accumulation is by node id (one ``g<id>`` slot per node), so a forward-shared (hash-consed) node has a
single adjoint slot and its rule fires once -- the same global, content-addressed shared-adjoint property
the oracle has, preserved verbatim in the emitted C (a shared node is one ``v<id>``/``g<id>`` pair). The
gradient w.r.t. each parameter is then read from the adjoint slot of that parameter's ``var`` node.

THE SGD STEP. ``emit_sgd_step_c`` emits the minimal deployable update ``w[i] -= lr * grad[i]`` over the
parameter vector -- one step is the primitive; a few steps reduce a loss (shown in the tests). It is a
separate ``void`` function so it composes with any gradient source. ``emit_train_step_c`` wires
grad-kernel -> SGD into one ``void train_step(float *params, float lr, float *value_out)`` so a caller can
run a full loop in C.

CLOSED PRIMITIVE SET ONLY. The emitter lowers EXACTLY the primitives ``autodiff`` supports --
``neg/add/sub/mul/div/dot/select/exp/log/sqrt/tanh/sin/cos`` plus the ``const``/``var``
leaves -- and raises deterministically on anything outside it. Transcendentals use the
host's declared libm edge. This mirrors the oracle's closure property and is the same
closed set ``autodiff._BACKWARD`` covers.

This is the ORACLE realization + emitter + verification (the reference reverse-mode grad, the
finite-difference gate, the compile-and-run parity). Wiring it into an MLIR ``gem.grad`` / backward-pass
law op is the separate G6-MLIR follow-up (kept separate, exactly as inference.py / matmul.py / activation.py
keep their MLIR wiring separate -- DO NOT touch mlir/ here).
"""

from __future__ import annotations

from ..kbcir.autodiff import Tape, _BACKWARD, _topo_order

# The closed primitive set this emitter lowers: the differentiable ops (each carries a local _BACKWARD
# rule) plus the two leaves. This is EXACTLY autodiff's primitive set; anything else is rejected.
_MATH_OPS = frozenset({"exp", "log", "sqrt", "tanh", "sin", "cos"})
_LOWERABLE = (
    frozenset({"const", "var", "neg", "add", "sub", "mul", "div", "dot", "select"}) | _MATH_OPS
)


def _fc(v) -> str:
    """A C float32 literal for ``v``, round-trip-exact (``%.9g``) and ALWAYS float-suffixed: a bare integer
    print like ``1`` would make ``1f`` invalid (a float suffix needs a ``.``/``e``), so force a ``.0`` when
    the text has no decimal point or exponent. The value is unchanged. (Same rule as inference._fc.)"""
    s = f"{float(v):.9g}"
    if not any(ch in s for ch in ".eEnN"):
        s += ".0"
    return s + "f"


def _validate(tape: Tape, order) -> None:
    """Reject any node outside the closed primitive set (deterministic). The oracle's closure -- every
    primitive's VJP lives in the same registered vocabulary -- is exactly what makes a straight-line
    backward lowering possible; a node outside the set has no emittable local rule, so we raise."""
    for nid in order:
        op = tape.node(nid).op
        if op not in _LOWERABLE:
            raise ValueError(
                f"autodiff_kernel: op {op!r} (node {nid}) is outside the closed lowerable "
                f"primitive set {sorted(_LOWERABLE)}; cannot emit a backward rule for it"
            )


def _param_index(tape: Tape, order, params) -> dict:
    """Map each requested parameter name -> its index in the emitted ``params[]`` vector. The order of
    ``params`` is the ABI: the caller's ``params[i]`` feeds the var named ``params[i]`` and ``grad_out[i]``
    is its gradient. A parameter that is NOT a var reachable from the output is allowed -- its gradient is
    identically 0 (an unused input has zero gradient), EXACTLY as ``autodiff.grad`` returns 0 for it; the
    emit reads ``grad_out[i] = 0.0f`` for such a param. (A duplicate parameter name IS an error -- the ABI
    index would be ambiguous.)"""
    if len(set(params)) != len(params):
        raise ValueError(
            f"autodiff_kernel: duplicate parameter name in {list(params)} (the params[] ABI "
            f"index would be ambiguous)"
        )
    return {name: i for i, name in enumerate(params)}


def _forward_expr(tape: Tape, nid: int, name_to_pidx: dict) -> str:
    """The C right-hand side computing node ``nid``'s forward value from its operands' ``v<id>`` temporaries
    -- the exact C translation of ``autodiff.evaluate`` / ``_forward_values`` for this op. A ``var`` reads
    its slot in the ``params[]`` vector; a ``const`` is a baked float literal."""
    n = tape.node(nid)
    op = n.op
    a = n.args
    if op == "const":
        return _fc(n.const)
    if op == "var":
        return f"params[{name_to_pidx[n.const]}]"
    if op == "neg":
        return f"-v{a[0]}"
    if op == "add":
        return f"v{a[0]} + v{a[1]}"
    if op == "sub":
        return f"v{a[0]} - v{a[1]}"
    if op == "mul":
        return f"v{a[0]} * v{a[1]}"
    if op == "div":
        return f"v{a[0]} / v{a[1]}"
    if op in ("exp", "log", "sqrt", "tanh", "sin", "cos"):
        return f"{op}f(v{a[0]})"
    if op == "select":
        cond, x, y = a
        # the SIGN TEST of cond's forward value (JAX lax.select / C ?:), identical to autodiff.evaluate.
        return f"(v{cond} > 0.0f ? v{x} : v{y})"
    if op == "dot":
        k = n.const[1]
        us, vs = a[:k], a[k:]
        if not us:
            return "0.0f"
        return " + ".join(f"v{u} * v{v}" for u, v in zip(us, vs))
    raise ValueError(f"autodiff_kernel: cannot emit forward for op {op!r}")  # pragma: no cover


def _needed_adjoints(tape: Tape, order, output: int, param_nids: set) -> set:
    """The node ids whose adjoint slot ``g<id>`` is actually READ -- so only those need a declared
    accumulator (keeping the emit warning-free under ``-Wall -Wextra``: no written-but-unread leaf slot).
    A slot ``g<id>`` is read iff:
      * node ``id`` is a NON-LEAF (its own backward rule reads ``g<id>`` as the incoming adjoint ``gz``), or
      * node ``id`` is a requested-parameter ``var`` (read into ``grad_out``).
    A ``const`` leaf or a non-parameter ``var`` leaf has its slot WRITTEN (accumulated into) but never read
    -- a differentiation boundary -- so it is excluded and its inbound contributions are elided. Dropping
    those is observationally identical (the slot feeds nothing), and is exactly the oracle's behavior: a
    leaf has no ``_BACKWARD`` rule, so its accumulated adjoint is dead. This NEVER drops a contribution that
    reaches a parameter gradient -- every node on a path to a parameter var is itself non-leaf or that var.

    NOTE on the ``output`` seed: when the output is a NON-LEAF, it is already in the set (it reads its own
    seed in its backward rule). When the output is a leaf VAR that is a requested param, it is in
    ``param_nids`` -> the seed 1.0 is read into ``grad_out`` (grad of x w.r.t. x = 1). When the output is a
    leaf CONST, NOTHING reads its seed (a const has zero gradient and no rule), so it is correctly excluded
    -- no dead ``g<output>`` is declared (keeps the degenerate const-output case warning-free).
    """
    needed = {nid for nid in order if tape.node(nid).op not in ("const", "var")}
    needed |= param_nids
    return needed


def _backward_lines(tape: Tape, nid: int, needed: set) -> list[str]:
    """The C accumulation lines for node ``nid``'s local backward (adjoint) rule -- the C translation of
    ``autodiff._BACKWARD[op]``. Each line is a ``g<operand> += <contribution>;`` into an operand's adjoint
    slot, reading forward values ``v<id>`` and the incoming adjoint ``g<nid>``. A leaf (``const``/``var``)
    has no rule (it is a differentiation boundary) and contributes nothing. A contribution into an operand
    whose slot is NOT in ``needed`` (a never-read leaf) is elided -- observationally identical, and what
    keeps the emit warning-free (see :func:`_needed_adjoints`).

    The contributions are EXACTLY the oracle's rules:
      neg:  g_a += -g_z
      add:  g_a += g_z ; g_b += g_z
      sub:  g_a += g_z ; g_b += -g_z
      mul:  g_a += g_z*v_b ; g_b += g_z*v_a
      div:  g_a += g_z/v_b ; g_b += -g_z*v_a/(v_b*v_b)
      dot:  g_{u_i} += g_z*v_{v_i} ; g_{v_i} += g_z*v_{u_i}
      select: routes g_z to the SELECTED branch only (a.e. convention), 0 to the other and to cond.
    """
    n = tape.node(nid)
    op = n.op
    a = n.args
    gz = f"g{nid}"

    def acc(operand: int, rhs: str) -> str | None:
        """One ``g<operand> += rhs;`` line, or None if that operand's slot is dead (never read)."""
        return f"  g{operand} += {rhs};" if operand in needed else None

    if op in ("const", "var"):
        return []  # a leaf -- no backward rule (boundary)
    if op == "neg":
        cand = [acc(a[0], f"-{gz}")]
    elif op == "add":
        cand = [acc(a[0], gz), acc(a[1], gz)]
    elif op == "sub":
        cand = [acc(a[0], gz), acc(a[1], f"-{gz}")]
    elif op == "mul":
        cand = [acc(a[0], f"{gz} * v{a[1]}"), acc(a[1], f"{gz} * v{a[0]}")]
    elif op == "div":
        b = a[1]
        cand = [acc(a[0], f"{gz} / v{b}"), acc(b, f"-{gz} * v{a[0]} / (v{b} * v{b})")]
    elif op == "exp":
        cand = [acc(a[0], f"{gz} * v{nid}")]
    elif op == "log":
        cand = [acc(a[0], f"{gz} / v{a[0]}")]
    elif op == "sqrt":
        cand = [acc(a[0], f"{gz} / (2.0f * v{nid})")]
    elif op == "tanh":
        cand = [acc(a[0], f"{gz} * (1.0f - v{nid} * v{nid})")]
    elif op == "sin":
        cand = [acc(a[0], f"{gz} * cosf(v{a[0]})")]
    elif op == "cos":
        cand = [acc(a[0], f"-{gz} * sinf(v{a[0]})")]
    elif op == "dot":
        k = n.const[1]
        us, vs = a[:k], a[k:]
        cand = []
        for u, v in zip(us, vs):
            cand.append(acc(u, f"{gz} * v{v}"))
            cand.append(acc(v, f"{gz} * v{u}"))
    elif op == "select":
        cond, x, y = a
        # the a.e. VJP: the incoming adjoint flows ENTIRELY to the selected branch, 0 to the other and to
        # cond. A single guarded statement is the C twin of _bw_select's branch (the boundary delta dropped,
        # matching JAX/PyTorch). cond gets +0 -> never accumulated. If neither selectable branch is needed
        # (both dead), the whole guard is elided.
        xline = f"g{x} += {gz}" if x in needed else None
        yline = f"g{y} += {gz}" if y in needed else None
        if xline is None and yline is None:
            return []
        return [
            f"  if (v{cond} > 0.0f) {{ {xline + ';' if xline else ''} }} "
            f"else {{ {yline + ';' if yline else ''} }}"
        ]
    else:
        raise ValueError(f"autodiff_kernel: cannot emit backward for op {op!r}")  # pragma: no cover
    return [ln for ln in cand if ln is not None]


def emit_autodiff_kernel_c(tape: Tape, output: int, params, fn_name: str = "bcir_grad") -> str:
    """Emit ONE self-contained C function

        void fn_name(const float *params, float *value_out, float *grad_out)

    that computes, for the DAG rooted at ``output`` over the ordered parameter list ``params`` (var names):
      * the FORWARD value -> ``value_out[0]``  (the C twin of ``autodiff.evaluate``), and
      * the reverse-mode BACKWARD gradients -> ``grad_out[i]`` = d(output)/d(params[i])  (the C twin of
        ``autodiff.grad``, applying each primitive's ``_BACKWARD`` rule in reverse topo order).

    The forward DAG's topological order (``autodiff._topo_order``) is BAKED into straight-line C: one
    ``float v<id>`` forward temporary per node and one ``float g<id>`` adjoint accumulator per node, the
    output adjoint seeded to 1, the operands' adjoints accumulated in reverse. A forward-shared
    (hash-consed) node is ONE ``v<id>``/``g<id>`` pair, so the content-addressed shared-adjoint property
    is preserved. ``params`` fixes the ABI: ``params[i]`` is the value of var ``params[i]`` and
    ``grad_out[i]`` is its gradient.

    Self-contained C: arithmetic graphs need only ``<stddef.h>``; graphs using the admitted
    transcendental vocabulary also include ``<math.h>`` and link through the platform-aware
    libm edge. The emit is warning-free under ``-Wall -Wextra`` and deterministic. Raises ``ValueError`` on any op outside
    the closed lowerable set, a duplicate parameter name, or a DAG var not bound by ``params``. A requested
    parameter that does not reach the output is allowed -- its gradient is identically 0 (the oracle's
    unused-input rule)."""
    params = tuple(params)
    order = _topo_order(tape, output)
    _validate(tape, order)
    name_to_pidx = _param_index(tape, order, params)

    # every var node in the DAG must be bound by params[] (the params vector IS the forward env, exactly as
    # autodiff.evaluate's env must bind every var); a var not in params is an unbound-input error.
    name_to_nid = {tape.node(nid).const: nid for nid in order if tape.node(nid).op == "var"}
    unbound = sorted(set(name_to_nid) - set(params))
    if unbound:
        raise ValueError(
            f"autodiff_kernel: var(s) {unbound} appear in the DAG but are not in params "
            f"(the params[] vector is the forward env -- every var must be bound)"
        )
    # the var node id carrying each reachable requested parameter (read into grad_out, a needed adjoint
    # slot). An unreachable param has no var node here -> its gradient is 0 (the oracle's unused-input rule).
    param_nids = {name_to_nid[name] for name in params if name in name_to_nid}
    needed = _needed_adjoints(tape, order, output, param_nids)

    n_params = len(params)
    n_nodes = len(order)
    # which ops actually appear (informational, for the attestation comment).
    ops_used = sorted({tape.node(nid).op for nid in order})

    needs_math = any(tape.node(nid).op in _MATH_OPS for nid in order)
    head = (
        f"/* BCIR -> G6 autodiff forward+backward kernel '{fn_name}' (Pillar 5b; lower the B3 autodiff\n"
        f" * oracle bcir.kbcir.autodiff to a self-contained reference-verified C kernel). Computes:\n"
        f" *   value_out[0] = forward value of the DAG   (C twin of autodiff.evaluate), and\n"
        f" *   grad_out[i]  = d(value)/d(params[i])       (reverse-mode: each primitive's _BACKWARD rule,\n"
        f" *                                               accumulated in reverse topo order).\n"
        f" * The forward DAG is fixed -> the topo order is BAKED to straight-line C (one v<id> forward temp\n"
        f" * + one g<id> adjoint accumulator per node; a hash-consed shared node is ONE pair -> the\n"
        f" * content-addressed shared-adjoint property is preserved). params[] order IS the ABI.\n"
        f" * nodes: {n_nodes}; params: {n_params}; primitives: {', '.join(ops_used)} (closed set --\n"
        f" *   neg/add/sub/mul/div/dot/select + const/var leaves; anything else is rejected).\n"
        f" * Reference-verified: the emitted gradients MATCH autodiff.grad AND finite_difference_grad to\n"
        f" * float round-off. scope: the deployable forward+backward+SGD training primitive -- NOT a full\n"
        f" * optimizer/data-loader/mini-batch framework (that breadth is deferred). */\n"
        "#include <stddef.h>\n" + ("#include <math.h>\n" if needs_math else "")
    )

    lines = [f"void {fn_name}(const float *params, float *value_out, float *grad_out) {{"]
    # If NO var reaches the output (a constant-valued DAG -> all gradients are 0), params is unreferenced;
    # mark it used so the degenerate case stays warning-free under -Wall -Wextra (-Wunused-parameter).
    if not name_to_nid:
        lines.append(
            "  (void)params;   /* no var reaches the output -> a constant DAG, all grads 0 */"
        )

    # --- the forward pass: one v<id> per node, in topo order (operands before users) ---
    lines.append(
        "  /* forward pass (topo order) -- the C twin of autodiff.evaluate / _forward_values. */"
    )
    for nid in order:
        lines.append(f"  float v{nid} = {_forward_expr(tape, nid, name_to_pidx)};")
    lines.append(f"  value_out[0] = v{output};")
    lines.append("")

    # --- the reverse pass: declare adjoint accumulators, seed the output adjoint, walk in reverse ---
    lines.append(
        "  /* reverse pass -- adjoint accumulators (one g<id> per NEEDED node; a never-read leaf"
    )
    lines.append(
        "     slot is elided -> warning-free), output adjoint seeded to 1, operand adjoints"
    )
    lines.append("     accumulated by each primitive's local _BACKWARD rule. */")
    for nid in order:
        if nid not in needed:
            continue  # a dead leaf slot -- never read, so not declared
        seed = "1.0f" if nid == output else "0.0f"
        lines.append(f"  float g{nid} = {seed};")
    # reverse topo order: users before operands (autodiff._reverse_order).
    lines.append("")
    for nid in reversed(order):
        if nid not in needed:
            continue  # a leaf (no rule) or a dead slot -> no firing
        lines.extend(_backward_lines(tape, nid, needed))
    lines.append("")

    # --- read off the gradient of each requested parameter from its adjoint slot (0 if unreachable) ---
    lines.append("  /* read the gradient w.r.t. each parameter from its var node's adjoint slot")
    lines.append(
        "     (an unreachable/unused input has identically-zero gradient, as the oracle). */"
    )
    for name in params:
        rhs = f"g{name_to_nid[name]}" if name in name_to_nid else "0.0f"
        lines.append(f"  grad_out[{name_to_pidx[name]}] = {rhs};")
    lines.append("}")
    return head + "\n" + "\n".join(lines) + "\n"


def emit_sgd_step_c(n_params: int, fn_name: str = "bcir_sgd_step") -> str:
    """Emit the minimal deployable SGD weight-update step

        void fn_name(float *params, const float *grad, float lr)   ->   params[i] -= lr * grad[i]

    over the length-``n_params`` parameter vector. ONE step is the primitive (a few reduce a loss -- shown
    in the tests). Kept a separate ``void`` so it composes with any gradient source. No libm; warning-free.
    This is the deployable update primitive: gradient descent's single move, the C twin of ``w -= lr*g``."""
    if n_params < 1:
        raise ValueError(f"sgd: n_params must be >= 1; got {n_params}")
    return (
        f"/* BCIR -> G6 minimal SGD weight-update step (the deployable update primitive). "
        f"params[i] -= lr * grad[i] over {n_params} params -- one step; a few steps reduce a loss. */\n"
        "#include <stddef.h>\n"
        f"void {fn_name}(float *params, const float *grad, float lr) {{\n"
        f"  for (size_t i = 0; i < {n_params}u; ++i)\n"
        f"    params[i] -= lr * grad[i];\n"
        f"}}\n"
    )


def emit_train_step_c(
    tape: Tape,
    output: int,
    params,
    grad_fn: str = "bcir_grad",
    sgd_fn: str = "bcir_sgd_step",
    fn_name: str = "bcir_train_step",
) -> str:
    """Emit the grad-kernel + SGD step + a wiring function

        void fn_name(float *params, float lr, float *value_out)

    that runs ONE training step in C: forward+backward (``grad_fn``) to get the loss value and gradients,
    then the SGD update (``sgd_fn``) ``params[i] -= lr*grad[i]``. ``value_out[0]`` is the loss BEFORE the
    update, so a caller looping over this sees the loss curve. This bundles the deployable training
    primitive into one self-contained translation unit (the grad kernel, the SGD step, and the loop body)."""
    params = tuple(params)
    n = len(params)
    grad_src = emit_autodiff_kernel_c(tape, output, params, grad_fn)
    sgd_src = emit_sgd_step_c(n, sgd_fn)
    wiring = (
        f"/* BCIR -> G6 one training step: forward+backward then SGD (the deployable loop body). "
        f"value_out[0] is the loss BEFORE the update -> a caller's loop sees the loss curve. */\n"
        f"void {fn_name}(float *params, float lr, float *value_out) {{\n"
        f"  float grad[{n}];\n"
        f"  {grad_fn}(params, value_out, grad);\n"
        f"  {sgd_fn}(params, grad, lr);\n"
        f"}}\n"
    )
    # grad_src + sgd_src each carry their own #include <stddef.h>; harmless duplicate (idempotent include).
    return grad_src + "\n" + sgd_src + "\n" + wiring


# --- compile + run the emitted kernels against the Python oracle (the gate that matters) -------------


def _which(name: str) -> str | None:
    from shutil import which

    return which(name)


def compile_and_run_grad_c(
    tape: Tape,
    output: int,
    params,
    env: dict,
    fn_name: str = "bcir_grad",
    workdir: str | None = None,
) -> tuple[bool, float, list[float], str]:
    """Emit the forward+backward kernel + a tiny ``main`` that runs it at the point ``env``, compile, run,
    and return ``(ok, value, grads, combined_output)`` where ``value`` is the forward value and ``grads``
    is the gradient list in ``params`` order. The caller compares ``value`` to ``autodiff.evaluate`` and
    ``grads`` to BOTH ``autodiff.grad`` and ``autodiff.finite_difference_grad`` (to float round-off).
    Arithmetic-only graphs need no libm; transcendental graphs use the platform-aware
    math-library link edge."""
    import os
    import subprocess
    import tempfile

    cc = _which("clang") or _which("cc") or _which("gcc")
    if cc is None:
        return False, 0.0, [], "no C compiler on PATH"
    params = tuple(params)
    kernel = emit_autodiff_kernel_c(tape, output, params, fn_name)
    pinit = ", ".join(_fc(env[name]) for name in params)
    n = len(params)
    main = (
        f"\n#include <stdio.h>\nint main(void) {{\n"
        f"  float params[{n}] = {{{pinit}}};\n"
        f"  float value; float grad[{n}];\n"
        f"  {fn_name}(params, &value, grad);\n"
        f'  printf("%.9g\\n", value);\n'
        f'  for (int i = 0; i < {n}; ++i) printf("%.9g ", grad[i]);\n'
        f"  return 0;\n}}\n"
    )

    created = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="bcir-grad-")
    try:
        src = os.path.join(workdir, "grad.c")
        exe = os.path.join(workdir, "grad")
        with open(src, "w") as f:
            f.write(kernel + main)
        needs_math = any(tape.node(nid).op in _MATH_OPS for nid in _topo_order(tape, output))
        build = None
        for std in ("-std=c23", "-std=c2x", "-std=c11"):
            from ..toolchain import host_link_args

            command = [cc, std, "-O2", "-Wall", "-Wextra", src, "-o", exe]
            if needs_math:
                command.append("-lm")
            build = subprocess.run(host_link_args(command), capture_output=True, text=True)
            if build.returncode == 0:
                break
        if build is None or build.returncode != 0:
            return (
                False,
                0.0,
                [],
                "grad build failed:\n" + (build.stdout + build.stderr if build else ""),
            )
        run = subprocess.run([exe], capture_output=True, text=True)
        if run.returncode != 0:
            return False, 0.0, [], "grad run failed:\n" + run.stdout + run.stderr
        toks = run.stdout.split()
        value = float(toks[0])
        grads = [float(t) for t in toks[1 : 1 + n]]
        return True, value, grads, run.stdout + run.stderr
    finally:
        if created:
            import shutil

            shutil.rmtree(workdir, ignore_errors=True)


def compile_and_run_training_c(
    tape: Tape,
    output: int,
    params,
    env: dict,
    lr: float,
    steps: int,
    fn_name: str = "bcir_train_step",
    workdir: str | None = None,
) -> tuple[bool, list[float], list[float], str]:
    """Emit the full train-step unit (grad kernel + SGD + wiring) + a ``main`` that runs ``steps`` training
    steps from the initial point ``env`` at learning rate ``lr``, and return ``(ok, losses, final_params,
    combined_output)``. ``losses[k]`` is the loss BEFORE step k (so a strictly decreasing list proves the
    SGD loop reduces the loss); ``final_params`` is the parameter vector after the last step.

    The caller runs the SAME loop in the oracle (forward via ``autodiff.evaluate``, backward via
    ``autodiff.grad``, then ``w -= lr*g``) and asserts the two loss curves AGREE -- the compiled C training
    loop matches the Python oracle, and both strictly decrease the loss."""
    import os
    import subprocess
    import tempfile

    cc = _which("clang") or _which("cc") or _which("gcc")
    if cc is None:
        return False, [], [], "no C compiler on PATH"
    params = tuple(params)
    n = len(params)
    unit = emit_train_step_c(tape, output, params, fn_name=fn_name)
    pinit = ", ".join(_fc(env[name]) for name in params)
    main = (
        f"\n#include <stdio.h>\nint main(void) {{\n"
        f"  float params[{n}] = {{{pinit}}};\n"
        f"  float lr = {_fc(lr)};\n"
        f"  float value;\n"
        f"  for (int s = 0; s < {int(steps)}; ++s) {{\n"
        f"    {fn_name}(params, lr, &value);\n"
        f'    printf("L %.9g\\n", value);\n'
        f"  }}\n"
        f'  for (int i = 0; i < {n}; ++i) printf("P %.9g\\n", params[i]);\n'
        f"  return 0;\n}}\n"
    )

    created = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="bcir-train-")
    try:
        src = os.path.join(workdir, "train.c")
        exe = os.path.join(workdir, "train")
        with open(src, "w") as f:
            f.write(unit + main)
        build = None
        for std in ("-std=c23", "-std=c2x", "-std=c11"):
            build = subprocess.run(
                [cc, std, "-O2", "-Wall", "-Wextra", src, "-o", exe], capture_output=True, text=True
            )
            if build.returncode == 0:
                break
        if build is None or build.returncode != 0:
            return (
                False,
                [],
                [],
                "train build failed:\n" + (build.stdout + build.stderr if build else ""),
            )
        run = subprocess.run([exe], capture_output=True, text=True)
        if run.returncode != 0:
            return False, [], [], "train run failed:\n" + run.stdout + run.stderr
        losses, fparams = [], []
        for ln in run.stdout.split("\n"):
            ln = ln.strip()
            if ln.startswith("L "):
                losses.append(float(ln[2:]))
            elif ln.startswith("P "):
                fparams.append(float(ln[2:]))
        return True, losses, fparams, run.stdout + run.stderr
    finally:
        if created:
            import shutil

            shutil.rmtree(workdir, ignore_errors=True)


# --- the oracle-side training loop (the reference the compiled C must match) -------------------------


def oracle_train(
    tape: Tape, output: int, params, env: dict, lr: float, steps: int
) -> tuple[list, dict]:
    """The reference SGD training loop, run purely in the Python oracle: ``steps`` iterations of
    forward (``autodiff.evaluate``) -> backward (``autodiff.grad``) -> SGD update ``w -= lr*g``, from the
    initial point ``env``. Returns ``(losses, final_env)`` where ``losses[k]`` is the loss BEFORE step k
    (so a strictly decreasing list proves the loop reduces the loss). This is the EXACT loop the compiled
    ``compile_and_run_training_c`` runs; the two loss curves must agree to float round-off."""
    from ..kbcir.autodiff import evaluate, grad

    cur = dict(env)
    losses = []
    for _ in range(int(steps)):
        loss = evaluate(tape, output, cur)
        losses.append(loss)
        g = grad(tape, output, cur).grads
        for name in params:
            cur[name] = cur[name] - lr * g[name]
    return losses, cur
