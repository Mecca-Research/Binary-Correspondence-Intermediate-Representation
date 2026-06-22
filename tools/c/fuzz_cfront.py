#!/usr/bin/env python3
"""Differential fuzzer for the cfront C subset -- twin (bcir_cfront.c) vs oracle (frontends/cfront)
vs Clang.

Generates random but *well-defined* C programs over the subset the two rails share, then for each:

  1. compares the total-compile OUTCOME (clean / dirty / fallback) of the two rails -- a construct one rail
     accepts while the other routes away is a subset divergence;
  2. for a mutually-clean unit, compares the structural claim SUMMARY (parity) and, if a C compiler is
     present, that BOTH rails' emitted C is behaviour-equivalent to Clang compiling the source.

GRAMMAR.  An optional prelude of helper functions, then an entry `f`. Both have 2-4 scalar parameters that
are each `unsigned` or `int`; `f` may additionally take one read-only `const unsigned *` parameter. Bodies
draw from arithmetic / bitwise / bounded shifts / comparisons / ternary / `if` / bounded `for` / statement
expressions / inc-dec, with `int` and `unsigned` mixed, same-unit calls into the helpers, and (for `f`)
bounded pointer reads `p[e & 7u]` / `*p`.

WELL-DEFINEDNESS (so Clang is a sound oracle -- every generated program is UB-free by construction):
  * `unsigned` arithmetic wraps, so `+ - * & | ^ << >>` are always defined (shifts are masked `& 31u`);
  * `int` arithmetic is the only overflow hazard, so the generator tracks a static absolute-value BOUND for
    every `int` expression (params are bounded by ±PMAX, which the driver respects) and forms a signed
    `+ - *` / `-x` / `~x` only when the result provably stays within INT range -- otherwise the operands are
    laundered through `unsigned` (defined). `int` locals are immutable after initialisation, so their bound
    is preserved; signed `<<` (UB on overflow/negative) is never formed, only the arithmetic `>>`;
  * there is no `/` or `%`; every local is initialised at its declaration; loops are bounded and never mutate
    their counter; an embedded value statement-expression mutates only locals it declares (no unsequenced
    side effect leaks into the enclosing expression); pointer reads are masked into a fixed backing array;
  * calls form a 2-level DAG (only `f` calls helpers; helpers call nothing) so there is no recursion, and
    `int` arguments are leaves bounded by ±PMAX so each callee's bound assumption holds.

Usage:  python tools/c/fuzz_cfront.py --count 400 --seed 0 [--verbose]
Exit status is nonzero on the first divergence (the offending source is printed)."""
from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_C = os.path.join(_ROOT, "runtime", "c")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_SAFE = (1 << 31) - 1           # INT_MAX: a signed result is only formed when its bound stays <= this
_PMAX = 1 << 14                 # the max |value| the driver feeds an `int` parameter (and an `int` leaf)
_ARRSZ = 8                      # the backing array length for a pointer parameter (reads masked `& 7u`)
_CMP = ["<", ">", "<=", ">=", "==", "!=", "&&", "||"]


class E:
    """A generated expression: its C text, its type ("u" unsigned / "i" int), and -- for `int` -- a static
    bound on |value| at runtime (0 for unsigned, which never overflows)."""
    __slots__ = ("text", "ty", "bound")

    def __init__(self, text: str, ty: str, bound: int = 0):
        self.text, self.ty, self.bound = text, ty, bound


def _iconst(c: int) -> str:
    return str(c) if c >= 0 else f"({c})"


class Gen:
    """A recursive, type-tracking generator with lexical-scope tracking (a name is only referenced where it
    is live) and signed-overflow-avoiding bound tracking. One instance emits one whole program."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.counter = 0
        self.helpers: list = []          # (name, [param-types], ret-type, ret-bound) for same-unit calls
        # per-function state (reset by _enter_function):
        self.scopes: list = [[]]
        self.ty: dict = {}               # name -> "u" | "i"
        self.bound: dict = {}            # name -> |value| bound (for "i" names)
        self.loopvars: set = set()       # `for` counters: readable, never a mutation target
        self.locked: set = set()         # names an embedded value stmt-expr must not mutate
        self.params: list = []           # this function's parameter names (int args must be leaves <= PMAX)
        self.ptr = None                  # the `const unsigned *` parameter name, or None
        self.allow_calls = False

    # -- scope / leaf helpers --------------------------------------------------------------------------
    def _live(self) -> list:
        return [n for s in self.scopes for n in s]

    def _live_ty(self, ty: str) -> list:
        return [n for s in self.scopes for n in s if self.ty.get(n) == ty and n != self.ptr]

    def _targets(self) -> list:
        # an assignable name: a LIVE, UNSIGNED, non-counter, non-locked, non-pointer name. `int` names are
        # immutable (so their static bound is preserved); loop counters stay monotone (so loops terminate);
        # locked names belong to an enclosing expression (mutating them would be an unsequenced side effect).
        return [n for s in self.scopes for n in s
                if self.ty.get(n) == "u" and n not in self.loopvars and n not in self.locked and n != self.ptr]

    def _fresh(self, ty: str, bound: int = 0) -> str:
        self.counter += 1
        n = f"v{self.counter}"
        self.scopes[-1].append(n)
        self.ty[n], self.bound[n] = ty, bound
        return n

    def _u(self, e: E) -> str:
        return e.text if e.ty == "u" else f"(unsigned)({e.text})"

    # -- expressions: PURE (no mutation outside an embedded stmt-expr's own locals) --------------------
    def aexpr(self, depth: int) -> E:
        return self.iexpr(depth) if self.rng.random() < 0.45 else self.uexpr(depth)

    def uexpr(self, depth: int) -> E:
        """An expression whose type is `unsigned` -- always defined (wraparound)."""
        r = self.rng
        if depth <= 0 or r.random() < 0.34:
            return self._uleaf(depth)
        k = r.random()
        if k < 0.34:
            return E(f"({self._u(self.aexpr(depth - 1))} {r.choice(['+', '-', '*', '&', '|', '^'])} "
                     f"{self._u(self.aexpr(depth - 1))})", "u")
        if k < 0.48:
            return E(f"(({self._u(self.aexpr(depth - 1))}) {r.choice(['<<', '>>'])} "
                     f"({r.randint(0, 255)}u & 31u))", "u")
        if k < 0.62:
            return E(f"((unsigned)({self.aexpr(depth - 1).text} {r.choice(_CMP)} {self.aexpr(depth - 1).text}))", "u")
        if k < 0.74:
            return E(f"({self.aexpr(depth - 1).text} ? {self._u(self.aexpr(depth - 1))} : "
                     f"{self._u(self.aexpr(depth - 1))})", "u")
        if k < 0.84:
            return E(f"({r.choice(['-', '~'])}{self._u(self.aexpr(depth - 1))})", "u")
        if k < 0.90:
            return self.value_stmt_expr(depth - 1, "u")
        call = self._call(depth, "u")
        return call if call is not None else self._uleaf(depth)

    def iexpr(self, depth: int) -> E:
        """An expression whose type is `int`, carrying a static |value| bound. A signed `+ - * - ~` is only
        formed when its bound stays within INT range; otherwise it falls back to a (safe) sub-expression."""
        r = self.rng
        if depth <= 0 or r.random() < 0.40:
            return self._ileaf()
        k = r.random()
        if k < 0.34:
            a, b = self.iexpr(depth - 1), self.iexpr(depth - 1)
            op = r.choice(["+", "-", "*"])
            rb = a.bound * b.bound if op == "*" else a.bound + b.bound
            if rb <= _SAFE:
                return E(f"({a.text} {op} {b.text})", "i", rb)
            return a
        if k < 0.46:
            a = self.iexpr(depth - 1)
            if r.random() < 0.5:
                return E(f"(-{a.text})", "i", a.bound)            # |-a| == |a| (a != INT_MIN since bound<=SAFE)
            return E(f"(~{a.text})", "i", a.bound + 1) if a.bound + 1 <= _SAFE else a
        if k < 0.58:
            a = self.iexpr(depth - 1)
            return E(f"({a.text} >> {r.randint(0, 31)})", "i", a.bound)   # arithmetic shift, |result| <= |a|
        if k < 0.72:
            return E(f"({self.aexpr(depth - 1).text} {r.choice(_CMP)} {self.aexpr(depth - 1).text})", "i", 1)
        if k < 0.84:
            a, b = self.iexpr(depth - 1), self.iexpr(depth - 1)
            return E(f"({self.aexpr(depth - 1).text} ? {a.text} : {b.text})", "i", max(a.bound, b.bound))
        if k < 0.90:
            return self.value_stmt_expr(depth - 1, "i")
        call = self._call(depth, "i")
        return call if call is not None else self._ileaf()

    def _uleaf(self, depth: int) -> E:
        r = self.rng
        if self.ptr is not None and depth > 0 and r.random() < 0.35:     # a bounded read of the pointer arg
            if r.random() < 0.5:
                return E(f"{self.ptr}[({self.uexpr(depth - 1).text}) & {_ARRSZ - 1}u]", "u")
            return E(f"(*{self.ptr})", "u")
        live = self._live_ty("u")
        if live and r.random() < 0.6:
            return E(r.choice(live), "u")
        return E(f"{r.randint(0, 255)}u", "u")

    def _ileaf(self) -> E:
        r = self.rng
        live = self._live_ty("i")
        if live and r.random() < 0.6:
            n = r.choice(live)
            return E(n, "i", self.bound.get(n, _PMAX))
        c = r.randint(-1000, 1000)
        return E(_iconst(c), "i", abs(c))

    def _int_arg(self) -> str:
        # a call's `int` argument must be bounded by PMAX (so the callee's bound assumption holds): use only
        # an `int` PARAMETER (bounded by the driver) or a small constant -- never a local or a call result.
        r = self.rng
        ipar = [n for n in self.params if self.ty.get(n) == "i"]
        if ipar and r.random() < 0.6:
            return r.choice(ipar)
        return _iconst(r.randint(-1000, 1000))

    def _call(self, depth: int, want: str):
        if not self.allow_calls:
            return None
        cands = [h for h in self.helpers if h[2] == want]
        if not cands:
            return None
        name, ptys, rty, rb = self.rng.choice(cands)
        args = [self.uexpr(depth - 1).text if pt == "u" else self._int_arg() for pt in ptys]
        return E(f"{name}({', '.join(args)})", rty, rb)

    def value_stmt_expr(self, depth: int, ty: str) -> E:
        """`({ <decls/stmts>; <value-expr>; })` -- its own scope; the last item is a pure expression of type
        `ty`. As an operand of a larger expression it may only mutate locals it declares itself (every outer
        name is locked), so it introduces no unsequenced side effect into the enclosing expression."""
        saved = self.locked
        self.locked = saved | set(self._live())
        self.scopes.append([])
        parts = [self.stmt(depth - 1, allow_block=False) for _ in range(self.rng.randint(0, 2))]
        val = self.uexpr(depth) if ty == "u" else self.iexpr(depth)
        parts.append(f"{val.text};")
        self.scopes.pop()
        self.locked = saved
        return E("({ " + " ".join(parts) + " })", ty, val.bound)

    # -- statements: the only mutation sites; each mutates at most one UNSIGNED local. Every recursive call
    # -- decrements `depth`, and `depth <= 0` precedes any nesting -- so generation always terminates.
    def stmt(self, depth: int, allow_block: bool = True) -> str:
        r = self.rng
        if not self._targets():                                 # nothing mutable in scope yet -> declare one
            e = self.uexpr(depth)                               # (initialiser BEFORE the name is in scope)
            n = self._fresh("u")
            return f"unsigned {n} = {e.text};"
        k = r.random()
        if k < 0.24:                                            # a fresh initialised local (`int` or unsigned)
            if r.random() < 0.4:                                # (initialiser generated BEFORE the name is in
                e = self.iexpr(depth)                           #  scope, so it can't read itself)
                n = self._fresh("i", e.bound)
                return f"int {n} = {e.text};"
            e = self.uexpr(depth)
            n = self._fresh("u")
            return f"unsigned {n} = {e.text};"
        if k < 0.46:                                            # plain assignment (to an unsigned target)
            return f"{r.choice(self._targets())} = {self.aexpr(depth).text};"
        if k < 0.60:                                            # compound assignment
            return f"{r.choice(self._targets())} {r.choice(['+', '-', '*', '&', '|', '^'])}= {self.aexpr(depth).text};"
        if k < 0.72:                                            # inc / dec (value discarded -- well defined)
            return f"{r.choice(self._targets())}{r.choice(['++', '--'])};"
        if depth <= 0:                                          # base case: no more nesting
            return f"{r.choice(self._targets())} = {self.aexpr(0).text};"
        if k < 0.80:                                            # a void statement expression
            self.scopes.append([])
            inner = self.stmt(depth - 1, allow_block=False) if r.random() < 0.5 else \
                f"if ({self.aexpr(depth - 1).text}) {{ {r.choice(self._targets())} = {self.aexpr(depth - 1).text}; }}"
            self.scopes.pop()
            return f"({{ {inner} }});"
        if not allow_block:                                     # inside a value stmt-expr: no blocks
            return f"{r.choice(self._targets())} = {self.aexpr(depth).text};"
        if k < 0.90:                                            # if / else
            self.scopes.append([])
            then = " ".join(self.stmt(depth - 1) for _ in range(r.randint(1, 2)))
            self.scopes.pop()
            out = f"if ({self.aexpr(depth - 1).text}) {{ {then} }}"
            if r.random() < 0.5:
                self.scopes.append([])
                els = " ".join(self.stmt(depth - 1) for _ in range(r.randint(1, 2)))
                self.scopes.pop()
                out += f" else {{ {els} }}"
            return out
        iv = f"i{self.counter}"                                 # a bounded for loop (counter never mutated
        self.counter += 1                                       #  in the body, so it always terminates)
        self.scopes.append([iv])
        self.ty[iv] = "u"
        self.loopvars.add(iv)
        body = " ".join(self.stmt(depth - 1) for _ in range(r.randint(1, 2)))
        self.loopvars.discard(iv)
        self.scopes.pop()
        return f"for (unsigned {iv} = 0u; {iv} < {r.randint(1, 8)}u; {iv}++) {{ {body} }}"

    # -- whole functions / program ---------------------------------------------------------------------
    def _enter_function(self, ptypes: list, has_ptr: bool):
        names = ["a", "b", "c", "d"][:len(ptypes)]
        self.scopes = [list(names)]
        self.ty = {n: t for n, t in zip(names, ptypes)}
        self.bound = {n: (_PMAX if t == "i" else 0) for n, t in zip(names, ptypes)}
        self.loopvars, self.locked = set(), set()
        self.params = list(names)
        self.ptr = "p" if has_ptr else None

    def _function(self, name: str, ptypes: list, ret: str, has_ptr: bool, allow_calls: bool):
        self._enter_function(ptypes, has_ptr)
        self.allow_calls = allow_calls
        names = list(self.scopes[0])
        body = [self.stmt(2) for _ in range(self.rng.randint(2, 5))]
        rv = self.uexpr(3) if ret == "u" else self.iexpr(3)
        body.append(f"return {rv.text};")
        decl = ", ".join(("int " if t == "i" else "unsigned ") + n for n, t in zip(names, ptypes))
        if has_ptr:
            decl += ", const unsigned *p"
        ret_c = "int" if ret == "i" else "unsigned"
        src = f"{ret_c} {name}({decl})\n{{\n  " + "\n  ".join(body) + "\n}\n"
        return src, rv.bound

    def program(self) -> "Program":
        r = self.rng
        self.helpers, helper_src = [], []
        for hi in range(r.randint(0, 3)):                       # a prelude of leaf helpers (no calls)
            ptypes = [r.choice(["u", "i"]) for _ in range(r.randint(1, 3))]
            ret = r.choice(["u", "i"])
            src, rb = self._function(f"g{hi}", ptypes, ret, has_ptr=False, allow_calls=False)
            helper_src.append(src)
            self.helpers.append((f"g{hi}", ptypes, ret, rb))
        fptypes = [r.choice(["u", "i"]) for _ in range(r.randint(2, 4))]
        fret = r.choice(["u", "i"])
        has_ptr = r.random() < 0.4
        fsrc, _ = self._function("f", fptypes, fret, has_ptr=has_ptr, allow_calls=True)
        return Program("\n".join(helper_src + [fsrc]), fptypes, fret, has_ptr)


class Program:
    __slots__ = ("source", "ptypes", "ret", "has_ptr")

    def __init__(self, source: str, ptypes: list, ret: str, has_ptr: bool):
        self.source, self.ptypes, self.ret, self.has_ptr = source, ptypes, ret, has_ptr


def _oracle_outcome(src: str):
    from bcir.frontends.cfront.pipeline import compile_with_fallback
    r = compile_with_fallback(src, check_clang=False)
    return (2 if r.needs_fallback else (0 if r.is_clean else 1)), r


def _oracle_summary_emit(src: str):
    from bcir.frontends.cfront import compile_unit
    from bcir.model import Domain
    r = compile_unit(src, check_clang=False)
    funcs = r.lowered.functions
    cl = funcs[next(reversed(funcs))].claims                    # the entry `f` is the last-defined function
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


_U_CANDS = [0, 1, 2, 7, 255, 256, 65535, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 123456789, 0xDEADBEEF, 3, 1000000]
_I_CANDS = [0, 1, -1, 2, -2, 7, -7, 100, -100, _PMAX, -_PMAX, 12345, -12345, 9999]


def _arg_vectors(prog: Program, nptr: int) -> list:
    """A deterministic set of type-correct argument vectors (one C literal per scalar parameter; the pointer
    parameter cycles through the backing arrays). `int` values stay within +-PMAX so the program is UB-free."""
    vecs = []
    for j in range(16):
        args = []
        for k, t in enumerate(prog.ptypes):
            if t == "u":
                args.append(f"{_U_CANDS[(j + 3 * k) % len(_U_CANDS)]}u")
            else:
                args.append(_iconst(_I_CANDS[(j + 2 * k) % len(_I_CANDS)]))
        if prog.has_ptr:
            args.append(f"ARR{j % nptr}")
        vecs.append(", ".join(args))
    return vecs


def _behaviour_ok(cc: str, prog: Program, emit: str, d: str, label: str) -> tuple[bool, str]:
    renamed = re.sub(r"\bf\b", "f_s", prog.source)              # the entry `f` -> `f_s`; helpers untouched
    arrays = ("static const unsigned ARR0[8] = {3u,140u,7u,0u,99u,1234567u,255u,42u};\n"
              "static const unsigned ARR1[8] = {0u,4294967295u,1u,77u,8u,65535u,256u,9u};\n")
    nptr = 2
    drv = ["int main(void){"]
    for vec in _arg_vectors(prog, nptr):
        drv.append(f"  if(bcir_f({vec}) != f_s({vec})) return 1;")
    drv.append("  return 0;}")
    harness = ("#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n"   # string.h: emits use memcpy
               + (arrays if prog.has_ptr else "") + renamed + "\n" + emit + "\n" + "\n".join(drv))
    cpath = os.path.join(d, f"{label}.c")
    epath = os.path.join(d, label)
    with open(cpath, "w") as fh:
        fh.write(harness)
    for std in ("c23", "c2x", "c17"):
        b = subprocess.run([cc, f"-std={std}", "-O2", cpath, "-o", epath], capture_output=True, text=True)
        if b.returncode == 0:
            break
    else:
        return False, f"{label} build failed:\n{b.stderr[:500]}"
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
        prog = Gen(rng).program()
        src = prog.source
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
            for tag, emit in (("twin", t_emit), ("oracle", o_emit)):
                ok, msg = _behaviour_ok(cc, prog, emit, d, f"{tag}{i}")
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
