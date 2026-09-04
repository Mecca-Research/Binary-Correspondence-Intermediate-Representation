"""C-frontend perf-regression guard (Phase 4, segment 4).

A deterministic compile-cost measure -- claims, inter-procedural plan leaves, IPO reuse -- with no
wall-clock timing, so the regression signal is stable on any host (unlike runtime perf, which is
bare-metal-gated in `bcir.perf_budget`). The guard's invariants are *self-relative* (scaling ratios
and the IPO property), not magic numbers, so ordinary lowering changes pass while a super-linear
blowup or a broken inter-procedural summary fails.
"""

from __future__ import annotations


def frontend_cost(source: str, **kwargs) -> dict:
    """The unit's compile cost: total claims across functions, the composite plan's leaf count, and
    the IPO reuse count. Deterministic for a given source (the regression baseline)."""
    from .pipeline import compile_unit

    r = compile_unit(source, check_clang=False, **kwargs)
    return {
        "claims": sum(len(lf.claims) for lf in r.lowered.functions.values()),
        "leaves": r.ipo.get("leaves", 0),
        "reused": r.ipo.get("reused", 0),
    }


def chain_program(n: int) -> str:
    """A function with `n` sequential statements -- the input for the linear-scaling invariant."""
    body = "".join(f"  x = x + {i}u;\n" for i in range(1, n + 1))
    return f"unsigned f(unsigned x){{\n{body}  return x;\n}}\n"


def repeated_call_program(k: int) -> str:
    """An entry that calls one helper `k` times -- the input for the IPO (plan-once) invariant."""
    expr = " + ".join("g(x)" for _ in range(k))
    return f"unsigned g(unsigned y){{ return y*2u; }}\nunsigned f(unsigned x){{ return {expr}; }}\n"
