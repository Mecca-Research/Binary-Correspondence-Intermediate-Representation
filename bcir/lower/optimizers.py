"""M2 (ML Tier-1, roadmap Pillar 5b / Phase B4): ADAPTIVE OPTIMIZERS -- generalize the minimal SGD step.

This is the second of the ML Tier-1 trio (M1 losses -> M2 optimizers -> M3 the training loop). M1 built the
loss head (``bcir.kbcir.losses``) on top of the B3 autodiff so a full forward -> loss -> backward -> parameter
gradients pass runs end-to-end; ``bcir.lower.autodiff_kernel`` already lowers the forward+backward+SGD-step to
C. But the ONLY weight-update primitive there is plain SGD (``emit_sgd_step_c``: ``params[i] -= lr*grad[i]``).
M2 adds the standard ADAPTIVE optimizer family -- **momentum (heavy-ball)**, **RMSprop**, **Adam** (with bias
correction) -- as a reference oracle (pure Python) + an emitted C step, reference-verified (C == reference to
float round-off) and shown to converge on a convex problem (a linear model under MSE, where the minimum is
known). The pattern MIRRORS ``autodiff_kernel.emit_sgd_step_c`` exactly: a ``void`` step over the parameter
vector + state buffers, compiled and run by a tempdir harness; the existing SGD module is NOT modified -- this
is a new module built on top.

THE UPDATE CONVENTIONS (stated exactly -- there are several conflicting conventions in the wild; these are the
ones used here, matching the PyTorch / the canonical-paper forms for the most-common defaults):

  * SGD (re-exposed for parity with the existing C SGD):
        p <- p - lr * g

  * MOMENTUM (heavy-ball; the "grad-into-velocity, then descend" convention -- Polyak heavy-ball, the form
    PyTorch uses with ``dampening=0``):
        v <- beta * v + g
        p <- p - lr * v
    So ``v`` accumulates a *decayed sum of gradients* (NOT a convex combination -- the ``(1-beta)`` factor is
    NOT applied to ``g`` here; that is the "EMA of gradients" convention used by some frameworks and is a
    different rule). With this convention the effective step size near a steady gradient is ~``lr/(1-beta)``,
    so a momentum lr is typically chosen smaller than the bare-SGD lr. ``beta=0`` recovers plain SGD exactly.

  * RMSPROP (the per-coordinate adaptive-learning-rate method; the squared-gradient EMA, Tieleman & Hinton):
        s <- beta * s + (1 - beta) * g^2
        p <- p - lr * g / (sqrt(s) + eps)
    ``s`` is an exponential moving average of the SQUARED gradient (a convex combination -- the ``(1-beta)``
    IS applied here). The per-coordinate denominator ``sqrt(s)+eps`` rescales each gradient by its recent RMS,
    so steep/flat coordinates get comparable step sizes. ``eps`` (default ``1e-8``) is the numerical-stability
    floor: it keeps the division finite even when ``s == 0`` (a coordinate whose gradient has been exactly 0).

  * ADAM (the bias-corrected first+second-moment method, Kingma & Ba 2015 -- with the explicit step count
    ``t`` for bias correction; this is THE distinctive feature, and it matters most in the first few steps):
        t <- t + 1                                  (1-based; the first call increments 0 -> 1)
        m <- beta1 * m + (1 - beta1) * g            (1st-moment EMA -- the mean of the gradient)
        v <- beta2 * v + (1 - beta2) * g^2          (2nd-moment EMA -- the uncentered variance)
        mhat <- m / (1 - beta1^t)                   (BIAS CORRECTION: m starts at 0, so it is biased toward 0
                                                     for small t; dividing by (1 - beta1^t) un-biases it)
        vhat <- v / (1 - beta2^t)                   (likewise for the second moment)
        p <- p - lr * mhat / (sqrt(vhat) + eps)
    Defaults: ``beta1=0.9, beta2=0.999, eps=1e-8``. The bias correction is why ``t`` is part of the state: at
    ``t=1`` the divisors are ``(1-beta1) = 0.1`` and ``(1-beta2) = 0.001``, which un-bias the freshly-started
    (zero-initialized) moments -- WITHOUT it the early steps would be far too small. ``t`` round-trips across
    steps as part of the state (the C step takes ``int *t`` and increments it in place).

WHICH OPTIMIZERS RIDE LIBM ``sqrtf`` (the honest dependency, like the activation / B5 ``c.call.libm:`` edge):

  * SGD and MOMENTUM are PURE ARITHMETIC (``+ - * /`` only) -- their emitted C needs only ``<stddef.h>``, NO
    ``<math.h>``, and links with NO ``-lm``. This keeps them on the same freestanding-friendly footing as the
    existing ``emit_sgd_step_c``.
  * RMSPROP and ADAM need a SQUARE ROOT (the ``sqrt(s)`` / ``sqrt(vhat)`` denominator), so their emitted C
    ``#include <math.h>`` and call ``sqrtf`` -- the ``c.call.libm:`` edge. The compile/run harness therefore
    links ``-lm`` for those two (and only those two). This is documented honestly here rather than hidden: it
    is the SAME trusted-libm boundary the transcendental activations / losses ride; the optimizer arithmetic
    around it (the EMAs, the bias correction) stays in plain float.

THE TWO-TRUTH NOTE (LangRef §13, the cost/optimization-side quarantine). An optimizer is a COST /
OPTIMIZATION-side object: it decides HOW to descend a (legal) loss landscape -- it is NEVER an R-law legality
verdict. This module touches no verifier and emits no ``Diagnostic`` (a test pins this). It sits exactly where
the losses sit: off the legality path, on the graded/cost side.

ONE STEP IS THE PRIMITIVE. Each ``*_step`` here (reference and emitted-C) is ONE update -- it returns the new
``(params, state)`` (reference) or updates the buffers in place (C). A few steps reduce a loss; the
convergence is *shown* in the tests (each optimizer drives a convex MSE to near its known minimum). Wiring the
repeated-step LOOP (the full optimizer.minimize over a model+loss, with the gradient source) is M3 -- this
module delivers the single deployable step, exactly as ``emit_sgd_step_c`` is the single SGD move.

Deterministic, pure-Python on the reference rail. The reference functions are side-effect-free: they take the
old ``(params, state)`` and RETURN the new ones (they never mutate their arguments), so a caller can replay or
branch. The emitted C is the in-place twin (it updates ``params`` + the state buffers), reference-verified by
``compile_and_run_optimizer_c`` (self-skips with no C compiler), exactly the harness discipline of
``autodiff_kernel.compile_and_run_training_c``.
"""

from __future__ import annotations

import math

# Re-exposed from the existing emitter so the module is the single optimizer surface (we do NOT redefine the C
# SGD step -- M2 must not modify autodiff_kernel; it reuses its emit_sgd_step_c verbatim).
from .autodiff_kernel import _fc, _which, emit_sgd_step_c  # noqa: F401  (re-exported on purpose)


# === reference update rules (pure Python, deterministic, side-effect-free -> new (params, state)) ========

def sgd_step(params, grads, lr):
    """One plain-SGD step: ``p <- p - lr*g`` over the parameter vector. Re-exposed for parity with the
    existing C SGD (``emit_sgd_step_c``); ``beta=0`` momentum is the same rule. Returns the NEW params list
    (the argument is not mutated)."""
    params = [float(p) for p in params]
    grads = [float(g) for g in grads]
    if len(params) != len(grads):
        raise ValueError(f"sgd_step: params/grads length mismatch ({len(params)} != {len(grads)})")
    return [p - lr * g for p, g in zip(params, grads)]


def momentum_step(params, grads, velocity, lr, beta=0.9):
    """One heavy-ball MOMENTUM step. Convention (Polyak heavy-ball / PyTorch dampening=0):

        v' = beta*v + g
        p' = p - lr*v'

    i.e. the raw gradient is accumulated into the decayed velocity (NO ``(1-beta)`` factor on ``g``), then the
    step descends along the updated velocity. ``beta=0`` recovers plain SGD exactly. Returns the new
    ``(params', velocity')`` (arguments not mutated). Pure arithmetic -- no libm."""
    params = [float(p) for p in params]
    grads = [float(g) for g in grads]
    velocity = [float(v) for v in velocity]
    n = len(params)
    if not (len(grads) == len(velocity) == n):
        raise ValueError(f"momentum_step: length mismatch params={n} grads={len(grads)} "
                         f"velocity={len(velocity)}")
    new_v = [beta * v + g for v, g in zip(velocity, grads)]
    new_p = [p - lr * v for p, v in zip(params, new_v)]
    return new_p, new_v


def rmsprop_step(params, grads, sq_avg, lr, beta=0.9, eps=1e-8):
    """One RMSPROP step. Convention (squared-gradient EMA, Tieleman & Hinton):

        s' = beta*s + (1-beta)*g^2
        p' = p - lr*g / (sqrt(s') + eps)

    ``s`` is the exponential moving average of the squared gradient (the ``(1-beta)`` IS applied here, a convex
    combination). The per-coordinate ``sqrt(s')+eps`` denominator rescales each gradient by its recent RMS;
    ``eps`` (default 1e-8) keeps the division finite when a coordinate's squared-grad average is 0. Returns the
    new ``(params', sq_avg')`` (arguments not mutated). Rides libm ``sqrtf`` in the emitted C."""
    params = [float(p) for p in params]
    grads = [float(g) for g in grads]
    sq_avg = [float(s) for s in sq_avg]
    n = len(params)
    if not (len(grads) == len(sq_avg) == n):
        raise ValueError(f"rmsprop_step: length mismatch params={n} grads={len(grads)} sq_avg={len(sq_avg)}")
    new_s = [beta * s + (1.0 - beta) * g * g for s, g in zip(sq_avg, grads)]
    new_p = [p - lr * g / (math.sqrt(s) + eps) for p, g, s in zip(params, grads, new_s)]
    return new_p, new_s


def adam_step(params, grads, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    """One ADAM step WITH BIAS CORRECTION (Kingma & Ba 2015). Convention:

        t'   = t + 1                       (1-based step count; the first call takes t=0 -> t'=1)
        m'   = beta1*m + (1-beta1)*g       (1st-moment EMA)
        v'   = beta2*v + (1-beta2)*g^2     (2nd-moment EMA)
        mhat = m' / (1 - beta1^t')         (bias correction -- the zero-initialized moments are biased toward 0
        vhat = v' / (1 - beta2^t')          for small t'; the divisors un-bias them, and matter most early)
        p'   = p - lr*mhat / (sqrt(vhat) + eps)

    Defaults beta1=0.9, beta2=0.999, eps=1e-8. ``t`` is part of the STATE (it drives the bias correction) and
    is incremented HERE, so the returned ``t'`` is the count AFTER this step (use ``t'`` to call the next).
    Returns the new ``(params', m', v', t')`` (arguments not mutated). Rides libm ``sqrtf`` in the emitted C.

    NOTE the order: ``t`` is incremented FIRST, so a fresh optimizer is initialized with ``t=0`` and the very
    first step uses ``t'=1`` (divisors ``1-beta1 = 0.1`` and ``1-beta2 = 0.001``)."""
    params = [float(p) for p in params]
    grads = [float(g) for g in grads]
    m = [float(x) for x in m]
    v = [float(x) for x in v]
    n = len(params)
    if not (len(grads) == len(m) == len(v) == n):
        raise ValueError(f"adam_step: length mismatch params={n} grads={len(grads)} m={len(m)} v={len(v)}")
    t_new = int(t) + 1
    bc1 = 1.0 - beta1 ** t_new          # bias-correction divisor for the 1st moment
    bc2 = 1.0 - beta2 ** t_new          # bias-correction divisor for the 2nd moment
    new_m = [beta1 * mi + (1.0 - beta1) * g for mi, g in zip(m, grads)]
    new_v = [beta2 * vi + (1.0 - beta2) * g * g for vi, g in zip(v, grads)]
    new_p = []
    for p, mi, vi in zip(params, new_m, new_v):
        mhat = mi / bc1
        vhat = vi / bc2
        new_p.append(p - lr * mhat / (math.sqrt(vhat) + eps))
    return new_p, new_m, new_v, t_new


# === emitted C step functions (mirror emit_sgd_step_c; in-place update of params + state buffers) ========

def emit_momentum_step_c(n_params: int, fn_name: str = "bcir_momentum_step") -> str:
    """Emit the heavy-ball MOMENTUM step as one self-contained C function

        void fn_name(float *params, const float *grad, float *velocity, float lr, float beta)

    updating, for each of the ``n_params`` coordinates, ``velocity[i] = beta*velocity[i] + grad[i]`` then
    ``params[i] -= lr*velocity[i]`` IN PLACE (the C twin of :func:`momentum_step`). Pure arithmetic -- only
    ``<stddef.h>`` (NO ``<math.h>``, NO ``-lm``); warning-free under ``-Wall -Wextra``. ``velocity`` is the
    caller-owned state buffer (zero-initialized for a fresh optimizer); the step round-trips it across calls."""
    if n_params < 1:
        raise ValueError(f"momentum: n_params must be >= 1; got {n_params}")
    return (
        f"/* BCIR -> M2 momentum (heavy-ball) step. v[i] = beta*v[i] + grad[i]; params[i] -= lr*v[i] over\n"
        f" * {n_params} params -- one step (a few reduce a loss). Convention: raw grad into the decayed\n"
        f" * velocity (no (1-beta) on grad), then descend along v. Pure arithmetic -- no libm, no -lm. */\n"
        "#include <stddef.h>\n"
        f"void {fn_name}(float *params, const float *grad, float *velocity, float lr, float beta) {{\n"
        f"  for (size_t i = 0; i < {n_params}u; ++i) {{\n"
        f"    velocity[i] = beta * velocity[i] + grad[i];\n"
        f"    params[i] -= lr * velocity[i];\n"
        f"  }}\n"
        f"}}\n"
    )


def emit_rmsprop_step_c(n_params: int, fn_name: str = "bcir_rmsprop_step") -> str:
    """Emit the RMSPROP step as one self-contained C function

        void fn_name(float *params, const float *grad, float *sq_avg, float lr, float beta, float eps)

    updating, per coordinate, ``sq_avg[i] = beta*sq_avg[i] + (1-beta)*grad[i]^2`` then
    ``params[i] -= lr*grad[i] / (sqrtf(sq_avg[i]) + eps)`` IN PLACE (the C twin of :func:`rmsprop_step`).

    RIDES LIBM ``sqrtf`` (the ``c.call.libm:`` edge): emits ``#include <math.h>`` and calls ``sqrtf``, so the
    compile/run harness links ``-lm`` (unlike the pure-arithmetic SGD/momentum). ``eps`` is the
    numerical-stability floor on the denominator. Warning-free under ``-Wall -Wextra``. ``sq_avg`` is the
    caller-owned state buffer (zero-initialized for a fresh optimizer)."""
    if n_params < 1:
        raise ValueError(f"rmsprop: n_params must be >= 1; got {n_params}")
    return (
        f"/* BCIR -> M2 RMSprop step. s[i] = beta*s[i] + (1-beta)*grad[i]^2;\n"
        f" * params[i] -= lr*grad[i] / (sqrtf(s[i]) + eps) over {n_params} params -- one step.\n"
        f" * RIDES libm sqrtf (the c.call.libm: edge) -> #include <math.h>, link -lm. eps is the\n"
        f" * stability floor on the denominator (a 0 squared-grad average never divides-by-zero). */\n"
        "#include <stddef.h>\n"
        "#include <math.h>\n"
        f"void {fn_name}(float *params, const float *grad, float *sq_avg, "
        f"float lr, float beta, float eps) {{\n"
        f"  for (size_t i = 0; i < {n_params}u; ++i) {{\n"
        f"    sq_avg[i] = beta * sq_avg[i] + (1.0f - beta) * grad[i] * grad[i];\n"
        f"    params[i] -= lr * grad[i] / (sqrtf(sq_avg[i]) + eps);\n"
        f"  }}\n"
        f"}}\n"
    )


def emit_adam_step_c(n_params: int, fn_name: str = "bcir_adam_step") -> str:
    """Emit the ADAM step (WITH bias correction) as one self-contained C function

        void fn_name(float *params, const float *grad, float *m, float *v, int *t,
                     float lr, float beta1, float beta2, float eps)

    that increments the step count ``*t`` (1-based), updates the first/second-moment EMAs ``m``/``v``, applies
    the bias correction ``m/(1-beta1^t)`` and ``v/(1-beta2^t)``, and steps
    ``params[i] -= lr*mhat / (sqrtf(vhat) + eps)`` IN PLACE (the C twin of :func:`adam_step`).

    ``t`` is an ``int*`` (the step count for bias correction) -- it is part of the STATE and is incremented
    once per call, BEFORE the bias divisors are formed (so the first call uses t=1). ``m``, ``v`` are the
    caller-owned moment buffers (zero-initialized for a fresh optimizer); they round-trip across steps along
    with ``*t``. The bias-correction powers ``beta1^t`` / ``beta2^t`` are computed with a small integer-power
    loop in PLAIN float (no libm ``powf`` needed); only the ``sqrtf`` denominator RIDES libm.

    RIDES LIBM ``sqrtf`` (the ``c.call.libm:`` edge): emits ``#include <math.h>``, link ``-lm`` (unlike the
    pure-arithmetic SGD/momentum). Warning-free under ``-Wall -Wextra``."""
    if n_params < 1:
        raise ValueError(f"adam: n_params must be >= 1; got {n_params}")
    return (
        f"/* BCIR -> M2 Adam step (WITH bias correction). *t increments (1-based); m,v are the 1st/2nd-moment\n"
        f" * EMAs; mhat=m/(1-beta1^t), vhat=v/(1-beta2^t); params[i] -= lr*mhat/(sqrtf(vhat)+eps) over\n"
        f" * {n_params} params -- one step. The bias correction (the t-dependent divisors) matters most in the\n"
        f" * first few steps. RIDES libm sqrtf (c.call.libm: edge) -> #include <math.h>, link -lm. The\n"
        f" * beta^t powers use a plain-float integer-power loop (no libm powf). */\n"
        "#include <stddef.h>\n"
        "#include <math.h>\n"
        f"void {fn_name}(float *params, const float *grad, float *m, float *v, int *t,\n"
        f"               float lr, float beta1, float beta2, float eps) {{\n"
        f"  *t += 1;\n"
        f"  /* bias-correction divisors 1 - beta^t, with beta^t via a plain integer-power loop (no powf). */\n"
        f"  float b1p = 1.0f, b2p = 1.0f;\n"
        f"  for (int k = 0; k < *t; ++k) {{ b1p *= beta1; b2p *= beta2; }}\n"
        f"  float bc1 = 1.0f - b1p;\n"
        f"  float bc2 = 1.0f - b2p;\n"
        f"  for (size_t i = 0; i < {n_params}u; ++i) {{\n"
        f"    m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];\n"
        f"    v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];\n"
        f"    float mhat = m[i] / bc1;\n"
        f"    float vhat = v[i] / bc2;\n"
        f"    params[i] -= lr * mhat / (sqrtf(vhat) + eps);\n"
        f"  }}\n"
        f"}}\n"
    )


# === the compile-and-run harness (mirror compile_and_run_training_c) =====================================

# the optimizers and what state each needs (which buffers, whether it rides libm, the emitter), so the harness
# + a caller can dispatch by name. ``libm`` -> link -lm; ``state`` -> the float state-buffer names (Adam also
# carries the int step count ``t``, handled specially).
_OPTIMIZERS = {
    "sgd":      {"emit": None,                "state": (),           "libm": False},
    "momentum": {"emit": emit_momentum_step_c, "state": ("velocity",), "libm": False},
    "rmsprop":  {"emit": emit_rmsprop_step_c,  "state": ("sq_avg",),   "libm": True},
    "adam":     {"emit": emit_adam_step_c,     "state": ("m", "v"),    "libm": True},
}


def reference_optimizer_trajectory(opt: str, params, grad_source, lr, steps, **hp):
    """Run ``steps`` reference optimizer steps from ``params`` with a fixed per-step gradient supplied by
    ``grad_source`` (a callable ``step_index, params -> grads`` -- a constant or a model-derived gradient), and
    return the parameter TRAJECTORY ``[params_0, params_1, ..., params_steps]`` (the initial point plus the
    point after each step). The reference twin of :func:`compile_and_run_optimizer_c`; the two must agree to
    float round-off. ``hp`` overrides the optimizer hyperparameters (beta / beta1 / beta2 / eps)."""
    if opt not in _OPTIMIZERS:
        raise ValueError(f"unknown optimizer {opt!r}; choose from {sorted(_OPTIMIZERS)}")
    cur = [float(p) for p in params]
    traj = [list(cur)]
    n = len(cur)
    velocity = [0.0] * n
    sq_avg = [0.0] * n
    m = [0.0] * n
    v = [0.0] * n
    t = 0
    for s in range(int(steps)):
        g = [float(x) for x in grad_source(s, cur)]
        if opt == "sgd":
            cur = sgd_step(cur, g, lr)
        elif opt == "momentum":
            cur, velocity = momentum_step(cur, g, velocity, lr, beta=hp.get("beta", 0.9))
        elif opt == "rmsprop":
            cur, sq_avg = rmsprop_step(cur, g, sq_avg, lr,
                                       beta=hp.get("beta", 0.9), eps=hp.get("eps", 1e-8))
        elif opt == "adam":
            cur, m, v, t = adam_step(cur, g, m, v, t, lr,
                                     beta1=hp.get("beta1", 0.9), beta2=hp.get("beta2", 0.999),
                                     eps=hp.get("eps", 1e-8))
        traj.append(list(cur))
    return traj


def compile_and_run_optimizer_c(opt: str, params, grads, lr, steps, *, workdir=None, **hp):
    """Emit the chosen optimizer's C step + a tiny ``main`` that runs ``steps`` updates from the initial
    ``params`` with a FIXED per-step gradient ``grads`` (a constant gradient vector, supplied every step), and
    return ``(ok, trajectory, combined_output)`` where ``trajectory`` is ``[params_0, ..., params_steps]`` --
    the C parameter trajectory used by the tests to prove C == reference.

    A fixed (constant) gradient is the cleanest parity fixture: it exercises the FULL state machinery
    (velocity / squared-grad EMA / Adam's m,v,t with bias correction round-tripping across steps) without
    needing the model gradient baked into C. Links ``-lm`` for rmsprop/adam (they ride libm ``sqrtf``); no
    ``-lm`` for the pure-arithmetic sgd/momentum. Self-skips (returns ``ok=False`` with a reason) if no C
    compiler is on PATH, exactly like ``compile_and_run_training_c``. ``hp`` overrides the hyperparameters."""
    import os
    import subprocess
    import tempfile

    if opt not in _OPTIMIZERS:
        raise ValueError(f"unknown optimizer {opt!r}; choose from {sorted(_OPTIMIZERS)}")
    spec = _OPTIMIZERS[opt]
    cc = _which("clang") or _which("cc") or _which("gcc")
    if cc is None:
        return False, [], "no C compiler on PATH"

    params = [float(p) for p in params]
    grads = [float(g) for g in grads]
    n = len(params)
    if len(grads) != n:
        raise ValueError(f"compile_and_run_optimizer_c: params/grads length mismatch ({n} != {len(grads)})")

    beta = hp.get("beta", 0.9)
    beta1 = hp.get("beta1", 0.9)
    beta2 = hp.get("beta2", 0.999)
    eps = hp.get("eps", 1e-8)
    pinit = ", ".join(_fc(p) for p in params)
    ginit = ", ".join(_fc(g) for g in grads)
    nsteps = int(steps)

    if opt == "sgd":
        step_src = emit_sgd_step_c(n, "bcir_sgd_step")
        decls = ""
        call = "    bcir_sgd_step(params, grad, lr);\n"
    elif opt == "momentum":
        step_src = emit_momentum_step_c(n, "bcir_momentum_step")
        decls = f"  float velocity[{n}] = {{0}};\n"
        call = f"    bcir_momentum_step(params, grad, velocity, lr, {_fc(beta)});\n"
    elif opt == "rmsprop":
        step_src = emit_rmsprop_step_c(n, "bcir_rmsprop_step")
        decls = f"  float sq_avg[{n}] = {{0}};\n"
        call = f"    bcir_rmsprop_step(params, grad, sq_avg, lr, {_fc(beta)}, {_fc(eps)});\n"
    else:  # adam
        step_src = emit_adam_step_c(n, "bcir_adam_step")
        decls = f"  float m[{n}] = {{0}}; float v[{n}] = {{0}}; int t = 0;\n"
        call = (f"    bcir_adam_step(params, grad, m, v, &t, lr, "
                f"{_fc(beta1)}, {_fc(beta2)}, {_fc(eps)});\n")

    main = (
        f"\n#include <stdio.h>\nint main(void) {{\n"
        f"  float params[{n}] = {{{pinit}}};\n"
        f"  const float grad[{n}] = {{{ginit}}};\n"
        f"  float lr = {_fc(lr)};\n"
        f"{decls}"
        f'  for (int i = 0; i < {n}; ++i) printf("P %.9g\\n", params[i]);\n'   # the initial point (step 0)
        f"  for (int s = 0; s < {nsteps}; ++s) {{\n"
        f"{call}"
        f'    for (int i = 0; i < {n}; ++i) printf("P %.9g\\n", params[i]);\n'
        f"  }}\n"
        f"  return 0;\n}}\n")

    created = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix=f"bcir-opt-{opt}-")
    try:
        src = os.path.join(workdir, "opt.c")
        exe = os.path.join(workdir, "opt")
        with open(src, "w") as f:
            f.write(step_src + main)
        link = ["-lm"] if spec["libm"] else []
        build = None
        for std in ("-std=c23", "-std=c2x", "-std=c11"):
            build = subprocess.run([cc, std, "-O2", "-Wall", "-Wextra", src, "-o", exe, *link],
                                   capture_output=True, text=True)
            if build.returncode == 0:
                break
        if build is None or build.returncode != 0:
            return False, [], "opt build failed:\n" + (build.stdout + build.stderr if build else "")
        run = subprocess.run([exe], capture_output=True, text=True)
        if run.returncode != 0:
            return False, [], "opt run failed:\n" + run.stdout + run.stderr
        flat = [float(ln[2:]) for ln in run.stdout.split("\n") if ln.strip().startswith("P ")]
        # flat is (steps+1) blocks of n params each -> reshape into the trajectory.
        traj = [flat[k * n:(k + 1) * n] for k in range(nsteps + 1)]
        return True, traj, run.stdout + run.stderr
    finally:
        if created:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)
