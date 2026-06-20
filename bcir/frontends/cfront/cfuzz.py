"""C-frontend fuzzing program (Phase 4, segment 4 — security/fuzzing).

Two campaigns harden the frontend:

  * `fuzz_valid` generates random programs from the *always-defined* unsigned subset (`+ - * & | ^`
    over `unsigned`, no division/shift, so there is no UB) -- now spanning **control flow and name
    scoping** as well as expressions: `for` loops that reuse a counter, bare `{ ... }` blocks, and a
    loop/block whose variable shadows a parameter that is read again after the construct. Each must
    lower, verify R1-R18 clean, and (with a C compiler) be Clang-equivalent. No crashes, no fallbacks,
    no mismatches -- so the campaign is a standing guard for the emit-compilation / scope-leak class of
    miscompiles (a verified unit whose emit does not compile, or computes a different value than Clang).
  * `fuzz_malformed` corrupts those programs byte-wise and feeds them through `compile_with_fallback`
    and `diagnose`, which must always *return* (the totality / never-crash contract), never raise.

Both are seeded, so a failing campaign reproduces from its seed.
"""
from __future__ import annotations

import random

_OPS = ("+", "-", "*", "&", "|", "^")          # total over unsigned; no UB (no /, %, <<, >>)
_INJECT = "(){};,+-*/&|^=<> ab1\"'"             # the corruption alphabet for the malformed campaign


def gen_expr(rng: random.Random, depth: int, names: list[str]) -> str:
    """A random arithmetic expression tree over the parameter names and small unsigned constants."""
    if depth <= 0 or rng.random() < 0.3:
        return rng.choice(names + [f"{rng.randint(0, 255)}u"])
    return f"({gen_expr(rng, depth - 1, names)} {rng.choice(_OPS)} {gen_expr(rng, depth - 1, names)})"


_CMPOUND = ("+=", "-=", "*=", "&=", "|=", "^=")    # total over unsigned (no /=, %=, shift-assign)


def gen_stmt(rng: random.Random, depth: int, names: list[str], params: list[str]) -> str:
    """A random *single* statement (braced where it expands to several, so it nests safely inside an
    `if`/`for` arm) that may update the accumulator `s`. The forms deliberately span the control / scope
    surface that single-expression programs never reached -- `for` loops that *reuse* a counter name,
    bare `{ ... }` blocks with a local, and -- the patterns that hid real miscompiles -- a loop/block
    whose variable *shadows a parameter* and where that parameter is read again *after* the construct
    (where a scope leak would wrongly resolve to the loop/block var). `names` is everything in scope (for
    expressions); shadow targets are drawn from `params` only, which stay function-scoped, so the
    post-construct read is always valid C. Stays in the always-defined unsigned subset (arithmetic +
    control flow; loop bounds are small constants, so every loop terminates and there is no UB), so a
    clean compiler is Clang-equivalent on every generated program."""
    r = rng.random()
    if depth <= 0 or r < 0.34:                                     # s OP= <expr>;
        return f"s {rng.choice(_CMPOUND)} {gen_expr(rng, rng.randint(1, 2), names)};"
    if r < 0.50:                                                   # a bare block with a local
        nm = f"v{rng.randint(0, 3)}"
        return (f"{{ unsigned {nm} = {gen_expr(rng, 2, names)}; "
                f"{gen_stmt(rng, depth - 1, names + [nm], params)} }}")
    if r < 0.66:                                                   # a for-loop, counter reused as i/j/k
        cv = rng.choice(("i", "j", "k"))
        return (f"for(unsigned {cv} = 0u; {cv} < {rng.randint(2, 6)}u; {cv}++) "
                f"{gen_stmt(rng, depth - 1, names + [cv], params)}")
    if r < 0.80:                                                   # shadow a param in a block, read the
        p = rng.choice(params)                                     # OUTER one after it (scope-leak bait)
        return (f"{{ {{ unsigned {p} = {gen_expr(rng, 2, names)}; s ^= {p}; }} s += {p}; }}")
    if r < 0.90:                                                   # shadow a param in a loop, read the
        p = rng.choice(params)                                     # OUTER one after it (scope-leak bait)
        return (f"{{ for(unsigned {p} = 0u; {p} < {rng.randint(2, 5)}u; {p}++) s += {p}; s -= {p}; }}")
    c = gen_expr(rng, 2, names)                                    # if/else
    return (f"if({c}) {gen_stmt(rng, depth - 1, names, params)} "
            f"else {gen_stmt(rng, depth - 1, names, params)}")


def gen_program(rng: random.Random, *, max_depth: int = 4) -> str:
    """A random, well-formed program from the verified unsigned-arithmetic + control-flow subset. Half
    are a bare expression return (the original campaign); half build an accumulator through a handful of
    statements -- loops, blocks, shadowing -- so the Clang-equivalence campaign exercises lowering, the
    emit, and (critically) name scoping, not just expression folding."""
    names = [chr(ord("a") + i) for i in range(rng.randint(1, 3))]
    sig = ", ".join(f"unsigned {n}" for n in names)
    if rng.random() < 0.5:                                         # the original single-expression form
        return f"unsigned f({sig}){{ return {gen_expr(rng, rng.randint(1, max_depth), names)}; }}\n"
    body = "  unsigned s = 0u;\n"                                  # the richer statement form
    for _ in range(rng.randint(2, 5)):
        body += "  " + gen_stmt(rng, rng.randint(1, 3), names, names) + "\n"
    return f"unsigned f({sig}){{\n{body}  return s + {gen_expr(rng, 2, names)};\n}}\n"


def mutate(src: str, rng: random.Random) -> str:
    """One random single-character corruption (delete / insert / replace) -- a malformed-input sample."""
    if not src:
        return src
    i = rng.randrange(len(src))
    kind = rng.choice(("del", "ins", "rep"))
    if kind == "del":
        return src[:i] + src[i + 1:]
    ch = rng.choice(_INJECT)
    return src[:i] + ch + (src[i:] if kind == "ins" else src[i + 1:])


def fuzz_valid(seed: int = 0, trials: int = 200, *, check_clang: bool = False) -> dict:
    """Compile random valid programs. `crash` / `fallback` / `mismatch` must all be 0."""
    from .pipeline import compile_with_fallback
    rng = random.Random(seed)
    stats = {"trials": trials, "fallback": 0, "mismatch": 0, "crash": 0}
    for _ in range(trials):
        src = gen_program(rng)
        try:
            r = compile_with_fallback(src, check_clang=check_clang)
        except Exception:  # noqa: BLE001 -- a crash is exactly what the campaign is hunting for
            stats["crash"] += 1
            continue
        if r.needs_fallback:
            stats["fallback"] += 1
        elif r.equivalence == "MISMATCH":
            stats["mismatch"] += 1
    return stats


def fuzz_malformed(seed: int = 0, trials: int = 400) -> dict:
    """Corrupt valid programs and run the front end -- it must always return, never raise (`crash` 0)."""
    from .pipeline import compile_with_fallback, diagnose
    rng = random.Random(seed)
    stats = {"trials": trials, "crash": 0}
    for _ in range(trials):
        src = gen_program(rng)
        for _ in range(rng.randint(1, 5)):
            src = mutate(src, rng)
        try:
            compile_with_fallback(src, check_clang=False)
            diagnose(src)
        except Exception:  # noqa: BLE001 -- the totality contract forbids this
            stats["crash"] += 1
    return stats
