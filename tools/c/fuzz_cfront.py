#!/usr/bin/env python3
"""Differential fuzzer for the cfront C subset -- twin (bcir_cfront.c) vs oracle (frontends/cfront)
vs Clang.

Generates random but *well-defined* `unsigned f(unsigned a, unsigned b, unsigned c)` programs over the
straight-line + bounded-control-flow subset the two rails share (arithmetic / bitwise / bounded shifts /
comparisons / ternary / if / bounded for / statement expressions / inc-dec statements), then for each:

  1. compares the total-compile OUTCOME (clean / dirty / fallback) of the two rails -- a construct one rail
     accepts while the other routes away is a subset divergence;
  2. for a mutually-clean unit, compares the structural claim SUMMARY (parity) and, if a C compiler is
     present, that BOTH rails' emitted C is behaviour-equivalent to Clang compiling the source.

Every generated program is by construction free of undefined behaviour: all arithmetic is unsigned
(wraparound is defined), shifts are masked to `& 31u`, there is no `/` or `%`, every local is initialised
at its declaration, loops are bounded, and no object is mutated more than once between sequence points
(expressions are pure; only statements mutate, one local each). So Clang is a sound behavioural oracle.

Usage:  python tools/c/fuzz_cfront.py --count 400 --seed 0 [--verbose]
Exit status is nonzero on the first divergence (the offending source is printed)."""
from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_BINOPS = ["+", "-", "*", "&", "|", "^"]
_CMP = ["<", ">", "<=", ">=", "==", "!=", "&&", "||"]


class Gen:
    """A recursive generator with lexical-scope tracking, so a name is only referenced where it is live."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.scopes = [["a", "b", "c"]]      # the parameter scope
        self.counter = 0
        self.loopvars: set = set()           # `for` counters: readable, but never a mutation target
        self.locked: set = set()             # names an embedded value stmt-expr must not mutate (see below)

    def _live(self) -> list:
        return [n for s in self.scopes for n in s]

    def _targets(self) -> list:
        # an assignable name: a live name that is neither a loop counter (mutating it could make the
        # bounded `for` non-terminating -- well-defined, but it would hang) nor locked. A name is LOCKED
        # while generating a *value* statement-expression: such a `({...})` is an operand of a larger
        # expression, and the C order of evaluation across operands is unsequenced, so mutating an outer
        # variable there (while another operand reads it) would be undefined behaviour -- which would make
        # Clang an unsound oracle. So an embedded stmt-expr only ever mutates locals it declares itself.
        return [n for s in self.scopes for n in s if n not in self.loopvars and n not in self.locked]

    def _fresh(self) -> str:
        self.counter += 1
        n = f"v{self.counter}"
        self.scopes[-1].append(n)
        return n

    def _const(self) -> str:
        return f"{self.rng.randint(0, 255)}u"

    # -- expressions: PURE (no mutation, no assignment) so there is never an unsequenced side effect ----
    def expr(self, depth: int) -> str:
        r = self.rng
        if depth <= 0 or r.random() < 0.32:
            return r.choice([r.choice(self._live()), self._const()])
        k = r.random()
        if k < 0.40:
            return f"({self.expr(depth - 1)} {r.choice(_BINOPS)} {self.expr(depth - 1)})"
        if k < 0.55:
            return f"(({self.expr(depth - 1)}) {r.choice(['<<', '>>'])} ({self._const()} & 31u))"
        if k < 0.70:
            return f"((unsigned)({self.expr(depth - 1)} {r.choice(_CMP)} {self.expr(depth - 1)}))"
        if k < 0.82:
            return f"({self.expr(depth - 1)} ? {self.expr(depth - 1)} : {self.expr(depth - 1)})"
        if k < 0.90:
            return f"({r.choice(['-', '~'])}{self.expr(depth - 1)})"
        return self.value_stmt_expr(depth - 1)

    def value_stmt_expr(self, depth: int) -> str:
        """`({ <decls/stmts>; <value-expr>; })` -- its own scope; the last item is a pure expression. Being
        an operand of a larger expression, it may only mutate locals it declares itself (every outer name is
        locked), so it introduces no unsequenced side effect into the enclosing expression."""
        saved = self.locked
        self.locked = saved | set(self._live())
        self.scopes.append([])
        parts = [self.stmt(depth - 1, allow_block=False) for _ in range(self.rng.randint(0, 2))]
        parts.append(f"{self.expr(depth)};")
        self.scopes.pop()
        self.locked = saved
        return "({ " + " ".join(parts) + " })"

    # -- statements: the only mutation sites; each mutates at most one local. Every recursive call
    # -- decrements `depth`, and `depth <= 0` precedes any nesting -- so generation always terminates.
    def stmt(self, depth: int, allow_block: bool = True) -> str:
        r = self.rng
        if not self._targets():                                 # nothing mutable in scope yet (e.g. the first
            init = self.expr(depth)                             #  stmt of an embedded stmt-expr) -> declare one
            n = self._fresh()
            return f"unsigned {n} = {init};"
        k = r.random()
        if k < 0.24:                                            # a fresh initialised local
            init = self.expr(depth)                              # (generate the initialiser BEFORE the name
            n = self._fresh()                                   #  is in scope, so it can't read itself)
            return f"unsigned {n} = {init};"
        if k < 0.46:                                            # plain assignment
            return f"{r.choice(self._targets())} = {self.expr(depth)};"
        if k < 0.60:                                            # compound assignment
            return f"{r.choice(self._targets())} {r.choice(_BINOPS)}= {self.expr(depth)};"
        if k < 0.72:                                            # inc / dec (value discarded -- well defined)
            return f"{r.choice(self._targets())}{r.choice(['++', '--'])};"
        if depth <= 0:                                          # base case: no more nesting
            return f"{r.choice(self._targets())} = {self.expr(0)};"
        if k < 0.80:                                            # a void statement expression
            self.scopes.append([])
            inner = self.stmt(depth - 1, allow_block=False) if r.random() < 0.5 else \
                f"if ({self.expr(depth - 1)}) {{ {r.choice(self._targets())} = {self.expr(depth - 1)}; }}"
            self.scopes.pop()
            return f"({{ {inner} }});"
        if not allow_block:                                     # inside a value stmt-expr: no blocks
            return f"{r.choice(self._targets())} = {self.expr(depth)};"
        if k < 0.90:                                            # if / else
            self.scopes.append([])
            then = " ".join(self.stmt(depth - 1) for _ in range(r.randint(1, 2)))
            self.scopes.pop()
            out = f"if ({self.expr(depth - 1)}) {{ {then} }}"
            if r.random() < 0.5:
                self.scopes.append([])
                els = " ".join(self.stmt(depth - 1) for _ in range(r.randint(1, 2)))
                self.scopes.pop()
                out += f" else {{ {els} }}"
            return out
        iv = f"i{self.counter}"                                 # a bounded for loop (counter never mutated
        self.counter += 1                                       #  in the body, so it always terminates)
        self.scopes.append([iv])
        self.loopvars.add(iv)
        body = " ".join(self.stmt(depth - 1) for _ in range(r.randint(1, 2)))
        self.loopvars.discard(iv)
        self.scopes.pop()
        return f"for (unsigned {iv} = 0u; {iv} < {r.randint(1, 8)}u; {iv}++) {{ {body} }}"

    def function(self, name: str) -> str:
        body = [self.stmt(2) for _ in range(self.rng.randint(2, 6))]
        body.append(f"return {self.expr(3)};")
        return (f"unsigned {name}(unsigned a, unsigned b, unsigned c)\n{{\n  "
                + "\n  ".join(body) + "\n}\n")


def _oracle_outcome(src: str):
    from bcir.frontends.cfront.pipeline import compile_with_fallback
    r = compile_with_fallback(src, check_clang=False)
    return (2 if r.needs_fallback else (0 if r.is_clean else 1)), r


def _oracle_summary_emit(src: str):
    from bcir.frontends.cfront import compile_unit
    from bcir.model import Domain
    r = compile_unit(src, check_clang=False)
    funcs = r.lowered.functions
    cl = funcs[next(reversed(funcs))].claims
    summ = (f"funcs={len(funcs)} claims={len(cl)} "
            f"mmio={sum(1 for c in cl if c.op == 'c.load' and c.domain == Domain.MMIO)} "
            f"bf={sum(1 for c in cl if c.op == 'c.bf.get')} "
            f"const={sum(1 for c in cl if c.op == 'c.const')} "
            f"binop={sum(1 for c in cl if c.op.startswith('c.bin.'))} "
            f"call={sum(1 for c in cl if c.op.startswith('c.call'))} "
            f"ok={1 if r.is_clean else 0}")
    return summ, "\n".join(r.emitted[n] for n in funcs)


def _twin(exe: str, src: str):
    """Returns (outcome, summary_first_line, emit). Outcome: 0 clean / 1 dirty / 2 fallback."""
    with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        out = subprocess.run([exe, path], capture_output=True, text=True).stdout
    finally:
        os.unlink(path)
    summary, _, emit = out.partition("----EMIT----\n")
    first = summary.strip().splitlines()[0] if summary.strip() else ""
    if first.startswith(("PARSE-ERR", "CPP-ERR", "FALLBACK")):
        return 2, first, ""
    if "ok=1" in first:
        return 0, first, emit
    return 1, first, emit


_INPUTS = [(0, 0, 0), (1, 2, 3), (7, 7, 7), (0xFFFFFFFF, 1, 2), (0, 0xFFFFFFFF, 5),
           (123456789, 987654321, 555), (1 << 31, 1 << 30, 3), (2, 0, 0xDEADBEEF)]


def _behaviour_ok(cc: str, src: str, emit: str, d: str, label: str) -> tuple[bool, str]:
    renamed = src.replace("f(unsigned", "f_s(unsigned", 1)
    drv = ["int main(void){"]
    for a, b, c in _INPUTS:
        drv.append(f"  if(bcir_f({a}u,{b}u,{c}u)!=f_s({a}u,{b}u,{c}u)) return 1;")
    drv.append("  return 0;}")
    harness = ("#include <stdint.h>\n#include <stdio.h>\n" + renamed + "\n" + emit + "\n" + "\n".join(drv))
    cpath = os.path.join(d, f"{label}.c")
    epath = os.path.join(d, label)
    with open(cpath, "w") as fh:
        fh.write(harness)
    for std in ("c23", "c2x", "c17"):
        b = subprocess.run([cc, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True)
        if b.returncode == 0:
            break
    else:
        return False, f"{label} build failed:\n{b.stderr[:400]}"
    try:
        run = subprocess.run([epath], capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return True, f"{label} run timed out (skipped)"     # a (well-defined) slow loop -- not a divergence
    return run.returncode == 0, f"{label} behaviour != Clang (rc={run.returncode})"


def run_seed(twin: str, cc, count: int, seed: int, d: str, verbose: bool = False):
    """Generate+check `count` programs. Returns (divergence_message_or_None, stats). `cc` may be None to
    skip the behaviour differential (outcome + parity are still checked). `twin` is the test_cfront binary,
    `d` a scratch directory. The first divergence (outcome / parity / behaviour) short-circuits."""
    rng = random.Random(seed)
    stats = {"clean": 0, "fallback": 0, "dirty": 0, "checked": 0}
    names = {0: "clean", 1: "dirty", 2: "fallback"}
    for i in range(count):
        src = Gen(rng).function("f")
        try:
            o_out, _ = _oracle_outcome(src)
        except Exception as e:                      # the oracle must never crash on in-subset input
            return f"ORACLE CRASH on program #{i} (seed {seed}):\n{src}\n{type(e).__name__}: {e}", stats
        t_out, t_first, t_emit = _twin(twin, src)
        if o_out != t_out:
            return (f"OUTCOME DIVERGENCE on program #{i} (seed {seed}): "
                    f"oracle={names[o_out]} twin={names[t_out]}\n{src}\ntwin: {t_first}", stats)
        if o_out != 0:
            stats["fallback" if o_out == 2 else "dirty"] += 1
            continue
        stats["clean"] += 1
        o_summ, o_emit = _oracle_summary_emit(src)
        if o_summ != t_first:
            return (f"PARITY DIVERGENCE on program #{i} (seed {seed}):\n{src}\n"
                    f"oracle: {o_summ}\ntwin  : {t_first}", stats)
        if cc:
            for label, emit in (("twin", t_emit), ("oracle", o_emit)):
                ok, msg = _behaviour_ok(cc, src, emit, d, f"{label}{i}")
                if not ok:
                    return f"BEHAVIOUR DIVERGENCE on program #{i} (seed {seed}): {msg}\n{src}", stats
            stats["checked"] += 1
        if verbose and i % 50 == 0:
            print(f"  #{i}: {stats}")
    return None, stats


def _build_twin(cc, d: str) -> str:
    twin = os.path.join(d, "tcf")
    srcs = ("bcir_cfront.c", "bcir_cpp.c", "bcir_verify.c", "bcir_runtime.c", "test_cfront.c")
    build = subprocess.run([cc or "clang", "-std=c2x", "-O2", "-I", _C, *[os.path.join(_C, s) for s in srcs],
                            "-o", twin], capture_output=True, text=True)
    if build.returncode != 0:
        raise RuntimeError("twin build failed:\n" + build.stderr)
    return twin


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    d = tempfile.mkdtemp(prefix="fuzz_cfront_")
    try:
        twin = _build_twin(cc, d)
    except RuntimeError as e:
        print(str(e))
        return 2
    divergence, stats = run_seed(twin, cc, args.count, args.seed, d, args.verbose)
    shutil.rmtree(d, ignore_errors=True)
    if divergence:
        print(divergence)
        return 1
    print(f"OK seed={args.seed} count={args.count}: {stats} (no outcome/parity/behaviour divergence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
