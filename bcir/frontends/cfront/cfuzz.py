"""C-frontend fuzzing program (Phase 4, segment 4 — security/fuzzing).

Two campaigns harden the frontend:

  * `fuzz_valid` generates random programs from the *always-defined* unsigned-arithmetic subset
    (`+ - * & | ^` over `unsigned`, no division/shift, so there is no UB) -- each must lower, verify
    R1-R18 clean, and (with a C compiler) be Clang-equivalent. No crashes, no fallbacks, no mismatches.
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


def gen_program(rng: random.Random, *, max_depth: int = 4) -> str:
    """A random, well-formed program from the verified unsigned-arithmetic subset."""
    names = [chr(ord("a") + i) for i in range(rng.randint(1, 3))]
    sig = ", ".join(f"unsigned {n}" for n in names)
    body = gen_expr(rng, rng.randint(1, max_depth), names)
    return f"unsigned f({sig}){{ return {body}; }}\n"


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
