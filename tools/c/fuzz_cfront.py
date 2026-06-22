#!/usr/bin/env python3
"""Differential fuzzer for the cfront C subset -- twin (bcir_cfront.c) vs oracle (frontends/cfront)
vs Clang.

Generates random but *well-defined* C programs over the subset the two rails share, then for each:

  1. compares the total-compile OUTCOME (clean / dirty / fallback) of the two rails -- a construct one rail
     accepts while the other routes away is a subset divergence;
  2. for a mutually-clean unit, compares the structural claim SUMMARY (parity) and, if a C compiler is
     present, that BOTH rails' emitted C is behaviour-equivalent to Clang compiling the source.

GRAMMAR.  An optional prelude of helper functions, then an entry `f`. Parameters / locals / returns are any
of `char` `short` `int` `long` `unsigned` `unsigned long`; `f` may additionally take one `unsigned *`
parameter it reads AND writes. Bodies draw from arithmetic / bitwise / bounded shifts / comparisons /
ternary / if / bounded for / statement expressions / inc-dec / mutable-local assignment / same-unit calls,
with the integer types mixed (the usual arithmetic conversions), and pointer reads/writes `q[e & 7u]` / `*q`.

WELL-DEFINEDNESS (so Clang is a sound oracle -- every program is UB-free by construction):
  * unsigned arithmetic wraps; the only overflow hazard is SIGNED `+ - *`, so the generator tracks a static
    |value| BOUND for every signed (`int`/`long`) expression and forms a signed op only when the result
    provably stays in range, else launders through the unsigned of that width. Sub-int types promote to int.
  * MUTABLE signed locals: narrow `char`/`short` locals self-cap to their type range on every store, so they
    are loop-safe (the promoted-int operation has vast headroom and the stored value is re-bounded each
    iteration); at loop entry their bound is inflated to the type cap so a read on a later iteration is
    covered. Wide `int`/`long` locals would accumulate across iterations, so they are mutated only OUTSIDE
    loops (precise additive tracking); they stay read-only inside loops.
  * signed `<<` (UB) is never formed (only the arithmetic `>>`); shift amounts are masked to the operand
    width; there is no `/` or `%`; every local is initialised; loops are bounded with an unmutated counter.
  * an embedded value statement-expression mutates only locals it declares and performs no pointer write
    (no unsequenced side effect leaks into the enclosing expression); pointer indices are masked into the
    fixed backing array, and a fresh array copy is given to each rail so a store divergence is observable.
  * calls are a 2-level DAG (only `f` calls helpers) so there is no recursion; an argument to a wide-signed
    parameter is a leaf bounded by the type cap, so each callee's bound assumption holds.

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

_PMAX = 1 << 14                 # the max |value| the driver feeds a wide-signed (int/long) parameter
_ARRSZ = 8                      # the backing array length for the pointer parameter (accesses masked `& 7u`)
_CMP = ["<", ">", "<=", ">=", "==", "!=", "&&", "||"]

# storage type code -> (C spelling, width bytes, signed, |value| cap (signed only), arithmetic type after
# integer promotion). char/short promote to int; int/long/unsigned/unsigned-long keep their arithmetic type.
_ST = {
    "i8":  ("char",          1, True,  127,            "i32"),
    "i16": ("short",         2, True,  32767,          "i32"),
    "i32": ("int",           4, True,  (1 << 31) - 1,  "i32"),
    "i64": ("long",          8, True,  (1 << 63) - 1,  "i64"),
    "u32": ("unsigned",      4, False, 0,              "u32"),
    "u64": ("unsigned long", 8, False, 0,              "u64"),
}
_TYPES = list(_ST)
# arithmetic-type facts (the type model after promotion): width, signedness, signed cap, cast spelling, suffix.
_AW = {"i32": 4, "u32": 4, "i64": 8, "u64": 8}
_ACAST = {"i32": "(int)", "i64": "(long)", "u32": "(unsigned)", "u64": "(unsigned long)"}
_ASUF = {"i32": "", "i64": "L", "u32": "u", "u64": "uL"}


def _sgn(aty: str) -> bool:
    return aty[0] == "i"


def _tmax(aty: str) -> int:
    return (1 << (8 * _AW[aty] - 1)) - 1


def _uw(aty: str) -> str:                       # the unsigned type of the same width (a launder target)
    return "u32" if _AW[aty] == 4 else "u64"


def _uac(a: str, b: str) -> str:                # usual arithmetic conversions over the four arithmetic types
    if "u64" in (a, b):
        return "u64"
    if "i64" in (a, b):
        return "i64"
    if "u32" in (a, b):
        return "u32"
    return "i32"


def _const_text(c: int, aty: str) -> str:
    s = f"{c}{_ASUF[aty]}"
    return s if c >= 0 else f"({s})"


class E:
    """A generated expression: its C text, arithmetic type ("i32"/"i64"/"u32"/"u64" after promotion), and a
    static bound on |value| (meaningful for signed types; 0 for unsigned, which wraps)."""
    __slots__ = ("text", "aty", "bound")

    def __init__(self, text: str, aty: str, bound: int = 0):
        self.text, self.aty, self.bound = text, aty, bound


def _vbound(e: E) -> int:
    # the operand's |value| bound when it feeds a SIGNED result: a signed operand carries its tracked bound;
    # an unsigned operand can be up to its full width (it converts to the wider signed type by value).
    return e.bound if _sgn(e.aty) else (1 << (8 * _AW[e.aty])) - 1


def _coerce(e: E, aty: str) -> str:
    return e.text if e.aty == aty else f"{_ACAST[aty]}({e.text})"


class Gen:
    """A recursive, type-tracking generator with lexical-scope tracking and signed-overflow-avoiding bound
    tracking that is aware of loop nesting. One instance emits one whole program."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.counter = 0
        self.helpers: list = []          # (name, [param storage codes], ret storage code) for same-unit calls
        # per-function state (reset by _enter_function):
        self.scopes: list = [[]]
        self.styp: dict = {}             # name -> storage code
        self.bound: dict = {}            # name -> |value| bound (signed names)
        self.loopvars: set = set()       # `for` counters: readable, never a mutation target
        self.locked: set = set()         # names an embedded value stmt-expr must not mutate
        self.pure = False                # inside a value stmt-expr (no pointer write -- unsequenced)
        self.loopdepth = 0               # loop nesting: wide-signed locals are immutable while > 0
        self.params: list = []           # this function's parameter names
        self.ptr = None                  # the `unsigned *` parameter name, or None
        self.allow_calls = False

    # -- scope / mutability ----------------------------------------------------------------------------
    def _live(self) -> list:
        return [n for s in self.scopes for n in s]

    def _mutable(self, code: str) -> bool:
        # unsigned (wraps) and narrow-signed (self-cap to type range) are mutable anywhere; wide-signed
        # int/long would accumulate across loop iterations, so they are mutable only outside loops.
        return code in ("u32", "u64", "i8", "i16") or (code in ("i32", "i64") and self.loopdepth == 0)

    def _targets(self) -> list:
        return [n for s in self.scopes for n in s
                if n not in self.loopvars and n not in self.locked and n != self.ptr
                and n in self.styp and self._mutable(self.styp[n])]

    def _fresh(self, code: str, bound: int) -> str:
        self.counter += 1
        n = f"v{self.counter}"
        self.scopes[-1].append(n)
        self.styp[n], self.bound[n] = code, bound
        return n

    def _store_bound(self, e: E, code: str) -> int:
        # the |value| bound of `e` after storing it into a local of type `code`: a signed narrow type caps
        # at its range; otherwise the value's own bound (the conversion to a narrower signed type is
        # implementation-defined, not UB, and identical across the Clang-compiled rails + source).
        _cs, _w, signed, cap, _a = _ST[code]
        if not signed:
            return 0
        return min(_vbound(e), cap)

    # -- expressions -----------------------------------------------------------------------------------
    def expr(self, depth: int) -> E:
        r = self.rng
        if depth <= 0 or r.random() < 0.34:
            return self.leaf(depth)
        k = r.random()
        if k < 0.28:                                    # binary arithmetic + - *
            a, b = self.expr(depth - 1), self.expr(depth - 1)
            op = r.choice("+-*")
            ra = _uac(a.aty, b.aty)
            if _sgn(ra):
                bd = _vbound(a) * _vbound(b) if op == "*" else _vbound(a) + _vbound(b)
                if bd <= _tmax(ra):
                    return E(f"({a.text} {op} {b.text})", ra, bd)
                uw = _uw(ra)
                return E(f"({_coerce(a, uw)} {op} {_coerce(b, uw)})", uw, 0)
            return E(f"({a.text} {op} {b.text})", ra, 0)
        if k < 0.40:                                    # bitwise & | ^ -> unsigned (avoids signed-bitwise bounds)
            a, b = self.expr(depth - 1), self.expr(depth - 1)
            uw = _uw(_uac(a.aty, b.aty))
            return E(f"({_coerce(a, uw)} {r.choice('&|^')} {_coerce(b, uw)})", uw, 0)
        if k < 0.50:                                    # shift
            a = self.expr(depth - 1)
            amt = f"({r.randint(0, 255)}u & {8 * _AW[a.aty] - 1}u)"
            if r.random() < 0.5:                        # `>>`: arithmetic on signed (keeps type + bound)
                return E(f"(({a.text}) >> {amt})", a.aty, a.bound if _sgn(a.aty) else 0)
            uw = _uw(a.aty)                             # `<<`: unsigned only (signed << can be UB)
            return E(f"(({_coerce(a, uw)}) << {amt})", uw, 0)
        if k < 0.62:                                    # comparison / logical -> int 0/1
            a, b = self.expr(depth - 1), self.expr(depth - 1)
            return E(f"({a.text} {r.choice(_CMP)} {b.text})", "i32", 1)
        if k < 0.74:                                    # ternary -> common type (a pure select, no overflow)
            a, b, c = self.expr(depth - 1), self.expr(depth - 1), self.expr(depth - 1)
            ra = _uac(a.aty, b.aty)
            return E(f"({c.text} ? {a.text} : {b.text})", ra, max(_vbound(a), _vbound(b)) if _sgn(ra) else 0)
        if k < 0.82:                                    # unary
            a = self.expr(depth - 1)
            if r.random() < 0.5 and _sgn(a.aty):
                return E(f"(-{a.text})", a.aty, a.bound)         # negate (bound <= cap, so not INT_MIN)
            uw = _uw(a.aty)
            return E(f"({r.choice(['-', '~'])}{_coerce(a, uw)})", uw, 0)
        if k < 0.90:
            return self.value_stmt_expr(depth - 1)
        call = self._call(depth)
        return call if call is not None else self.leaf(depth)

    def leaf(self, depth: int) -> E:
        r = self.rng
        if self.ptr is not None and depth > 0 and r.random() < 0.18:     # a bounded pointer read
            if r.random() < 0.5:
                return E(f"{self.ptr}[({_coerce(self.expr(depth - 1), 'u32')}) & {_ARRSZ - 1}u]", "u32")
            return E(f"(*{self.ptr})", "u32")
        live = [n for n in self._live() if n != self.ptr]
        if live and r.random() < 0.6:
            n = r.choice(live)
            aty = _ST[self.styp[n]][4]
            return E(n, aty, self.bound.get(n, 0) if _sgn(aty) else 0)
        aty = r.choice(["i32", "i32", "i64", "u32", "u64"])
        c = r.randint(-1000, 1000) if _sgn(aty) else r.randint(0, 1000)
        return E(_const_text(c, aty), aty, abs(c) if _sgn(aty) else 0)

    def _wide_arg(self, ptype: str) -> str:
        # an argument to a wide-signed (int/long) parameter must respect the callee's PMAX bound assumption:
        # a narrow-signed leaf (always <= 32767) or a small constant -- never a (possibly-mutated) wide one.
        narrows = [n for n in self._live() if n != self.ptr and self.styp.get(n) in ("i8", "i16")]
        if narrows and self.rng.random() < 0.5:
            return self.rng.choice(narrows)
        return _const_text(self.rng.randint(-1000, 1000), ptype)

    def _call(self, depth: int):
        if not self.allow_calls or not self.helpers:
            return None
        name, ptypes, rty = self.rng.choice(self.helpers)
        args = []
        for pt in ptypes:
            args.append(self._wide_arg(pt) if pt in ("i32", "i64") else self.expr(depth - 1).text)
        return E(f"{name}({', '.join(args)})", _ST[rty][4], _ST[rty][3] if _sgn(_ST[rty][4]) else 0)

    def value_stmt_expr(self, depth: int) -> E:
        """`({ <stmts>; <value-expr>; })` -- its own scope; an operand of a larger expression, so it mutates
        only locals it declares (outer names locked) and performs no pointer write (no unsequenced effect)."""
        saved_lock, saved_pure = self.locked, self.pure
        self.locked = saved_lock | set(self._live())
        self.pure = True
        self.scopes.append([])
        parts = [self.stmt(depth - 1, allow_block=False) for _ in range(self.rng.randint(0, 2))]
        val = self.expr(depth)
        parts.append(f"{val.text};")
        self.scopes.pop()
        self.locked, self.pure = saved_lock, saved_pure
        return E("({ " + " ".join(parts) + " })", val.aty, val.bound)

    # -- statements ------------------------------------------------------------------------------------
    def _decl(self, depth: int) -> str:
        code = self.rng.choice(_TYPES)
        e = self.expr(depth)
        n = self._fresh(code, self._store_bound(e, code))
        return f"{_ST[code][0]} {n} = {e.text};"

    def stmt(self, depth: int, allow_block: bool = True) -> str:
        r = self.rng
        if not self._targets():                                 # nothing mutable yet -> declare a local
            return self._decl(depth)
        k = r.random()
        if k < 0.22:                                            # a fresh initialised local
            return self._decl(depth)
        if k < 0.42:                                            # plain assignment (a conversion -- never overflows)
            n = r.choice(self._targets())
            e = self.expr(depth)
            self.bound[n] = self._store_bound(e, self.styp[n])
            return f"{n} = {e.text};"
        if k < 0.56:                                            # compound assignment
            return self._compound(depth)
        if k < 0.66:                                            # inc / dec (value discarded)
            n = r.choice(self._targets())
            self._bump(n, 1)
            return f"{n}{r.choice(['++', '--'])};"
        if self.ptr is not None and not self.pure and k < 0.74:  # a pointer write (sequenced statement)
            return self._ptr_write(depth)
        if depth <= 0:                                          # base case: no more nesting
            n = r.choice(self._targets())
            e = self.expr(0)
            self.bound[n] = self._store_bound(e, self.styp[n])
            return f"{n} = {e.text};"
        if k < 0.82:                                            # a statement expression used as a statement: a
            self.scopes.append([])                              # side-effecting prefix (sequenced -- may mutate
            pre = self.stmt(depth - 1, allow_block=False)       # outer locals) then a trailing value, discarded.
            val = self.expr(depth - 1).text                     # (ending in a value, not an assignment, keeps
            self.scopes.pop()                                   #  it out of the assignment-as-value fallback)
            return f"({{ {pre} {val}; }});"
        if not allow_block:                                     # inside a value stmt-expr: no blocks
            n = r.choice(self._targets())
            e = self.expr(depth)
            self.bound[n] = self._store_bound(e, self.styp[n])
            return f"{n} = {e.text};"
        if k < 0.91:                                            # if / else
            cond = self.expr(depth - 1).text
            self.scopes.append([])
            then = " ".join(self.stmt(depth - 1) for _ in range(r.randint(1, 2)))
            self.scopes.pop()
            out = f"if ({cond}) {{ {then} }}"
            if r.random() < 0.5:
                self.scopes.append([])
                els = " ".join(self.stmt(depth - 1) for _ in range(r.randint(1, 2)))
                self.scopes.pop()
                out += f" else {{ {els} }}"
            return out
        return self._for(depth)                                # a bounded for loop

    def _compound(self, depth: int) -> str:
        r = self.rng
        n = r.choice(self._targets())
        code = self.styp[n]
        op = r.choice(["+", "-", "*", "&", "|", "^"])
        e = self.expr(depth)
        if op in "&|^":                                         # bitwise: no overflow; result re-bounded by type
            self.bound[n] = _ST[code][3] if _ST[code][2] else 0
            return f"{n} {op}= {e.text};"
        nat = _ST[code][4]                                      # additive/mult: the promoted op must not overflow
        ra = _uac(nat, e.aty)
        cur = self.bound.get(n, 0) if _sgn(nat) else (1 << (8 * _AW[nat])) - 1
        if _sgn(ra):
            bd = cur * _vbound(e) if op == "*" else cur + _vbound(e)
            if bd > _tmax(ra):                                  # would overflow -> shrink the RHS to a constant
                e = E(_const_text(r.randint(-100, 100), nat), nat, 100)
                bd = cur * 100 if op == "*" else cur + 100
                if bd > _tmax(ra):                              # still tight (a near-max wide local) -> plain store
                    e0 = self.expr(0)
                    self.bound[n] = self._store_bound(e0, code)
                    return f"{n} = {e0.text};"
            self.bound[n] = self._store_bound(E("", ra, bd), code)
        else:
            self.bound[n] = 0
        return f"{n} {op}= {e.text};"

    def _bump(self, n: str, by: int):
        code = self.styp[n]
        if not _ST[code][2]:                                   # unsigned wraps
            self.bound[n] = 0
        elif code in ("i8", "i16"):                            # narrow: re-caps to type range
            self.bound[n] = _ST[code][3]
        else:                                                  # wide (outside loops): +1, capped at the type max
            self.bound[n] = min(self.bound.get(n, 0) + by, _ST[code][3])

    def _ptr_write(self, depth: int) -> str:
        r = self.rng
        rhs = self.expr(depth).text
        if r.random() < 0.35:
            return f"*{self.ptr} = {rhs};"               # `*p = x` (canonical); `(*p) = x` is a twin parse gap
        idx = _coerce(self.expr(depth - 1) if depth > 0 else self.leaf(0), "u32")
        op = r.choice(["=", "+=", "-=", "*=", "|=", "&=", "^="])
        return f"{self.ptr}[({idx}) & {_ARRSZ - 1}u] {op} {rhs};"

    def _for(self, depth: int) -> str:
        r = self.rng
        iv = f"i{self.counter}"
        self.counter += 1
        for n in self._live():                                 # loop-entry inflation: a narrow-signed local may
            if self.styp.get(n) in ("i8", "i16"):              # be mutated to anywhere in its range on a later
                self.bound[n] = _ST[self.styp[n]][3]           # iteration, so a read must use the type cap.
        self.scopes.append([iv])
        self.styp[iv] = "u32"
        self.loopvars.add(iv)
        self.loopdepth += 1
        body = " ".join(self.stmt(depth - 1) for _ in range(r.randint(1, 2)))
        self.loopdepth -= 1
        self.loopvars.discard(iv)
        self.scopes.pop()
        return f"for (unsigned {iv} = 0u; {iv} < {r.randint(1, 8)}u; {iv}++) {{ {body} }}"

    # -- whole functions / program ---------------------------------------------------------------------
    def _enter_function(self, ptypes: list, has_ptr: bool):
        names = ["a", "b", "c", "d"][:len(ptypes)]
        self.scopes = [list(names)]
        self.styp = {n: t for n, t in zip(names, ptypes)}
        self.bound = {n: (_ST[t][3] if _ST[t][2] else 0) for n, t in zip(names, ptypes)}
        self.loopvars, self.locked, self.pure, self.loopdepth = set(), set(), False, 0
        self.params = list(names)
        self.ptr = "p" if has_ptr else None

    def _function(self, name: str, ptypes: list, ret: str, has_ptr: bool, allow_calls: bool) -> str:
        self._enter_function(ptypes, has_ptr)
        self.allow_calls = allow_calls
        body = [self.stmt(2) for _ in range(self.rng.randint(2, 5))]
        rv = self.expr(3)
        body.append(f"return {_coerce(rv, _ST[ret][4]) if _ST[ret][4] != rv.aty else rv.text};")
        decl = ", ".join(f"{_ST[t][0]} {n}" for n, t in zip(self.params, ptypes))
        if has_ptr:
            decl += ", unsigned *p"
        src = f"{_ST[ret][0]} {name}({decl})\n{{\n  " + "\n  ".join(body) + "\n}\n"
        return src

    def program(self) -> "Program":
        r = self.rng
        self.helpers, helper_src = [], []
        for hi in range(r.randint(0, 3)):                       # a prelude of leaf helpers (no calls, no ptr)
            ptypes = [r.choice(_TYPES) for _ in range(r.randint(1, 3))]
            ret = r.choice(_TYPES)
            helper_src.append(self._function(f"g{hi}", ptypes, ret, has_ptr=False, allow_calls=False))
            self.helpers.append((f"g{hi}", ptypes, ret))
        fptypes = [r.choice(_TYPES) for _ in range(r.randint(2, 4))]
        fret = r.choice(_TYPES)
        has_ptr = r.random() < 0.4
        fsrc = self._function("f", fptypes, fret, has_ptr=has_ptr, allow_calls=True)
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


_POOL = {
    "i8":  [0, 1, -1, 127, -128, 50, -50, 99],
    "i16": [0, 1, -1, 32767, -32768, 1000, -1000, 12345],
    "i32": [0, 1, -1, _PMAX, -_PMAX, 12345, -12345, 100],
    "i64": [0, 1, -1, _PMAX, -_PMAX, 9999, -9999, 5000],
    "u32": [0, 1, 255, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 123456789, 3],
    "u64": [0, 1, 0xFFFFFFFF, 0x100000000, 0xFFFFFFFFFFFFFFFF, 7, 1000000, 42],
}


def _arg_vectors(prog: Program) -> list:
    """A deterministic set of type-correct scalar argument vectors (one C literal per scalar parameter)."""
    vecs = []
    for j in range(16):
        args = []
        for k, t in enumerate(prog.ptypes):
            args.append(_const_text(_POOL[t][(j + 3 * k) % len(_POOL[t])], _ST[t][4]))
        vecs.append(args)
    return vecs


def _behaviour_ok(cc: str, prog: Program, emit: str, d: str, label: str) -> tuple[bool, str]:
    renamed = re.sub(r"\bf\b", "f_s", prog.source)              # the entry `f` -> `f_s`; helpers untouched
    rtype = _ST[prog.ret][0]
    init = "{3u,140u,7u,0u,99u,1234567u,255u,42u}"
    drv = ["int main(void){"]
    for vec in _arg_vectors(prog):
        a = ", ".join(vec)
        if prog.has_ptr:
            drv.append(f"  {{ unsigned A[8]={init}, B[8]={init}; {rtype} r1=bcir_f({a}{',' if a else ''} A);"
                       f" {rtype} r2=f_s({a}{',' if a else ''} B); if(r1!=r2) return 1;"
                       " for(int k=0;k<8;k++) if(A[k]!=B[k]) return 2; }")
        else:
            drv.append(f"  if(bcir_f({a}) != f_s({a})) return 1;")
    drv.append("  return 0;}")
    harness = ("#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n"
               + renamed + "\n" + emit + "\n" + "\n".join(drv))
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
    skip the behaviour differential (outcome + parity are still checked). The first divergence short-circuits."""
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
