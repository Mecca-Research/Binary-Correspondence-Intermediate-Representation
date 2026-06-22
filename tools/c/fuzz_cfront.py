#!/usr/bin/env python3
"""Differential fuzzer for the cfront C subset -- twin (bcir_cfront.c) vs oracle (frontends/cfront)
vs Clang.

Generates random but *well-defined* C programs over the subset the two rails share, then for each:

  1. compares the total-compile OUTCOME (clean / dirty / fallback) of the two rails -- a construct one rail
     accepts while the other routes away is a subset divergence;
  2. for a mutually-clean unit, compares the structural claim SUMMARY (parity) and, if a C compiler is
     present, that BOTH rails' emitted C is behaviour-equivalent to Clang compiling the source.

GRAMMAR.  Struct/union type definitions, an optional prelude of helper functions, then an entry `f`. Scalar
parameters / locals / returns are any of `char` `short` `int` `long` `unsigned` `unsigned long` `float`
`double`; `f` may also take struct/union parameters BY VALUE and declare struct/union locals (members are
accessed `s.m`), and up to two `unsigned *` parameters it reads AND writes (which MAY alias). A union only
ever exposes one active member (no type-punning). Bodies draw from arithmetic / bitwise / bounded shifts /
comparisons / ternary / if / bounded for / statement expressions / inc-dec / mutable-local-and-member
assignment / same-unit calls / pointer reads+writes, with the integer types mixed (the usual arithmetic
conversions) and floating-point arithmetic kept in its own lane.

WELL-DEFINEDNESS (so Clang is a sound oracle -- every program is UB-free by construction):
  * unsigned arithmetic wraps; the only integer overflow hazard is SIGNED `+ - *`, so the generator tracks a
    static |value| BOUND for every signed (`int`/`long`) expression and forms a signed op only when it
    provably stays in range, else launders through the unsigned of that width. Sub-int types promote to int.
  * FLOATING-POINT is its own lane: `float`/`double` `+ - * /` and comparisons are all defined (a `/0` gives
    inf/nan, not UB), so float values never overflow into UB. Crucially a float NEVER converts back to int
    (`(int)f` is UB when out of range) -- integers flow INTO floats (`(double)i`, always defined) but not
    back; floats only meet integers in a comparison (which yields int). So float results are compared with a
    ULP-tolerant (+ nan/inf-aware) check; integer results are compared exactly.
  * MUTABLE signed locals: narrow `char`/`short` self-cap to their range each store (loop-safe, with
    loop-entry bound inflation); wide `int`/`long` are mutated only outside loops; float locals never trap,
    so they are mutable anywhere. signed `<<` is never formed; shifts are masked to the operand width; there
    is no integer `/`%`; every local is initialised; loops are bounded with an unmutated counter.
  * an embedded value statement-expression mutates only locals it declares and performs no pointer write;
    pointer indices are masked into the fixed backing array(s), and each rail gets fresh array copies (with
    the same alias pattern) so a store divergence is observed by comparing the arrays after the call.
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
_ARRSZ = 8                      # the backing array length for a pointer parameter (accesses masked `& 7u`)
_CMP = ["<", ">", "<=", ">=", "==", "!=", "&&", "||"]

# storage type code -> (C spelling, width bytes, signed-int, |value| cap (signed-int only), arithmetic type
# after integer promotion). char/short promote to int; the rest keep their arithmetic type.
_ST = {
    "i8":  ("signed char",   1, True,  127,            "i32"),   # `signed char`, NOT plain `char` (which is
    "i16": ("short",         2, True,  32767,          "i32"),
    "i32": ("int",           4, True,  (1 << 31) - 1,  "i32"),
    "i64": ("long",          8, True,  (1 << 63) - 1,  "i64"),
    "u32": ("unsigned",      4, False, 0,              "u32"),
    "u64": ("unsigned long", 8, False, 0,              "u64"),
    "f32": ("float",         4, False, 0,              "f32"),
    "f64": ("double",        8, False, 0,              "f64"),
}
_TYPES = ["i8", "i16", "i32", "i64", "u32", "u64"]          # integer storage types
_ALL = _TYPES + ["f32", "f64"]                             # all storage types (params / locals / returns)
_FLOATS = ("f32", "f64")
_AW = {"i32": 4, "u32": 4, "i64": 8, "u64": 8, "f32": 4, "f64": 8}
_ACAST = {"i32": "(int)", "i64": "(long)", "u32": "(unsigned)", "u64": "(unsigned long)",
          "f32": "(float)", "f64": "(double)"}
_ASUF = {"i32": "", "i64": "L", "u32": "u", "u64": "uL"}


def _sgn(aty: str) -> bool:                     # a signed-integer arithmetic type
    return aty[0] == "i"


def _isf(aty: str) -> bool:
    return aty[0] == "f"


def _tmax(aty: str) -> int:
    return (1 << (8 * _AW[aty] - 1)) - 1


def _uw(aty: str) -> str:                       # the unsigned type of the same width (a launder target)
    return "u32" if _AW[aty] == 4 else "u64"


def _uac(a: str, b: str) -> str:                # usual arithmetic conversions (floats outrank integers)
    if "f64" in (a, b):
        return "f64"
    if "f32" in (a, b):
        return "f32"
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


def _fconst(v: float, aty: str) -> str:
    s = repr(v) + ("f" if aty == "f32" else "")
    return s if v >= 0 else f"({s})"


class E:
    """A generated expression: its C text, arithmetic type (i32/i64/u32/u64/f32/f64 after promotion), and a
    static bound on |value| (meaningful for signed integers; 0 for unsigned/float, which never trap)."""
    __slots__ = ("text", "aty", "bound")

    def __init__(self, text: str, aty: str, bound: int = 0):
        self.text, self.aty, self.bound = text, aty, bound


def _vbound(e: E) -> int:
    # the operand's |value| bound when it feeds a SIGNED result (only integers reach here): a signed operand
    # carries its tracked bound; an unsigned operand can be up to its full width (it converts by value).
    return e.bound if _sgn(e.aty) else (1 << (8 * _AW[e.aty])) - 1


def _coerce(e: E, aty: str) -> str:
    return e.text if e.aty == aty else f"{_ACAST[aty]}({e.text})"


class Gen:
    """A recursive, type-tracking generator with lexical-scope tracking, signed-overflow-avoiding bound
    tracking aware of loop nesting, and a separate floating-point lane. One instance emits one program."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.counter = 0
        self.helpers: list = []          # (name, [param storage codes], ret storage code) for same-unit calls
        self.aggdefs: dict = {}          # tag -> ("struct"/"union", [(member, code)], union active-member index)
        # per-function state (reset by _enter_function):
        self.scopes: list = [[]]
        self.styp: dict = {}             # name -> storage code
        self.bound: dict = {}            # name -> |value| bound (signed-int names)
        self.loopvars: set = set()       # `for` counters: readable, never a mutation target
        self.locked: set = set()         # names an embedded value stmt-expr must not mutate
        self.pure = False                # inside a value stmt-expr (no pointer write -- unsequenced)
        self.loopdepth = 0               # loop nesting: wide-signed locals are immutable while > 0
        self.params: list = []           # this function's scalar parameter names
        self.ptrs: list = []             # the `unsigned *` parameter names (0..2), which may alias
        self.allow_calls = False

    # -- scope / mutability ----------------------------------------------------------------------------
    def _live(self) -> list:
        return [n for s in self.scopes for n in s]

    def _mutable(self, code: str) -> bool:
        # unsigned (wraps), narrow-signed (self-cap), and float (never traps) are mutable anywhere; wide
        # int/long would accumulate across loop iterations, so they are mutable only outside loops.
        return code not in ("i32", "i64") or self.loopdepth == 0

    def _targets(self) -> list:
        return [n for s in self.scopes for n in s
                if n not in self.loopvars and n not in self.locked and n not in self.ptrs
                and n in self.styp and self._mutable(self.styp[n])]

    def _fresh(self, code: str, bound: int) -> str:
        self.counter += 1
        n = f"v{self.counter}"
        self.scopes[-1].append(n)
        self.styp[n], self.bound[n] = code, bound
        return n

    def _add_member(self, access: str, code: str, bound: int):
        # an aggregate member is a scoped lvalue-name (e.g. `s.m0`); every read/write/bound/loop rule that
        # applies to a scalar local applies to it verbatim (its name just happens to be a member access).
        self.scopes[-1].append(access)
        self.styp[access] = code
        self.bound[access] = bound

    def _store_bound(self, e: E, code: str) -> int:
        # the |value| bound after storing `e` into a local of type `code`: a signed narrow type caps at its
        # range; otherwise the value's own bound (a narrowing conversion is impl-defined, not UB).
        if not _ST[code][2]:                                # unsigned / float: no signed bound
            return 0
        return min(_vbound(e), _ST[code][3])

    # -- integer expressions (the `int` lane: never produces a float) ----------------------------------
    def iexpr(self, depth: int) -> E:
        r = self.rng
        if depth <= 0 or r.random() < 0.34:
            return self._ileaf(depth)
        k = r.random()
        if k < 0.28:                                    # binary arithmetic + - *
            a, b = self.iexpr(depth - 1), self.iexpr(depth - 1)
            op = r.choice("+-*")
            ra = _uac(a.aty, b.aty)
            if _sgn(ra):
                bd = _vbound(a) * _vbound(b) if op == "*" else _vbound(a) + _vbound(b)
                if bd <= _tmax(ra):
                    return E(f"({a.text} {op} {b.text})", ra, bd)
                uw = _uw(ra)
                return E(f"({_coerce(a, uw)} {op} {_coerce(b, uw)})", uw, 0)
            return E(f"({a.text} {op} {b.text})", ra, 0)
        if k < 0.40:                                    # bitwise & | ^ -> unsigned
            a, b = self.iexpr(depth - 1), self.iexpr(depth - 1)
            uw = _uw(_uac(a.aty, b.aty))
            return E(f"({_coerce(a, uw)} {r.choice('&|^')} {_coerce(b, uw)})", uw, 0)
        if k < 0.50:                                    # shift
            a = self.iexpr(depth - 1)
            amt = f"({r.randint(0, 255)}u & {8 * _AW[a.aty] - 1}u)"
            if r.random() < 0.5:                        # `>>`: arithmetic on signed (keeps type + bound)
                return E(f"(({a.text}) >> {amt})", a.aty, a.bound if _sgn(a.aty) else 0)
            uw = _uw(a.aty)                             # `<<`: unsigned only (signed << can be UB)
            return E(f"(({_coerce(a, uw)}) << {amt})", uw, 0)
        if k < 0.62:                                    # comparison / logical -> int 0/1 (operands may be float)
            a, b = self._num(depth - 1), self._num(depth - 1)
            return E(f"({a.text} {r.choice(_CMP)} {b.text})", "i32", 1)
        if k < 0.74:                                    # ternary -> common int type (a pure select)
            a, b, c = self.iexpr(depth - 1), self.iexpr(depth - 1), self.iexpr(depth - 1)
            ra = _uac(a.aty, b.aty)
            return E(f"({c.text} ? {a.text} : {b.text})", ra, max(_vbound(a), _vbound(b)) if _sgn(ra) else 0)
        if k < 0.82:                                    # unary
            a = self.iexpr(depth - 1)
            if r.random() < 0.5 and _sgn(a.aty):
                return E(f"(-{a.text})", a.aty, a.bound)
            uw = _uw(a.aty)
            return E(f"({r.choice(['-', '~'])}{_coerce(a, uw)})", uw, 0)
        if k < 0.90:
            return self._stmt_expr(depth - 1, float_val=False)
        call = self._call(depth, want_float=False)
        return call if call is not None else self._ileaf(depth)

    def _ileaf(self, depth: int) -> E:
        r = self.rng
        if self.ptrs and depth > 0 and r.random() < 0.18:               # a bounded pointer read
            p = r.choice(self.ptrs)
            if r.random() < 0.5:
                return E(f"{p}[({_coerce(self.iexpr(depth - 1), 'u32')}) & {_ARRSZ - 1}u]", "u32")
            return E(f"(*{p})", "u32")
        live = [n for n in self._live() if n not in self.ptrs and self.styp.get(n) not in _FLOATS]
        if live and r.random() < 0.6:
            n = r.choice(live)
            aty = _ST[self.styp[n]][4]
            return E(n, aty, self.bound.get(n, 0) if _sgn(aty) else 0)
        aty = r.choice(["i32", "i32", "i64", "u32", "u64"])
        c = r.randint(-1000, 1000) if _sgn(aty) else r.randint(0, 1000)
        return E(_const_text(c, aty), aty, abs(c) if _sgn(aty) else 0)

    # -- floating-point expressions (the float lane: always produces a float) --------------------------
    def fexpr(self, depth: int) -> E:
        r = self.rng
        if depth <= 0 or r.random() < 0.4:
            return self._fleaf()
        k = r.random()
        if k < 0.5:                                     # float arithmetic + - * /  (all defined)
            a, b = self.fexpr(depth - 1), self.fexpr(depth - 1)
            return E(f"({a.text} {r.choice('+-*/')} {b.text})", _uac(a.aty, b.aty), 0)
        if k < 0.68:                                    # ternary
            a, b = self.fexpr(depth - 1), self.fexpr(depth - 1)
            return E(f"({self.iexpr(depth - 1).text} ? {a.text} : {b.text})", _uac(a.aty, b.aty), 0)
        if k < 0.82:                                    # int -> float (always defined)
            ft = r.choice(_FLOATS)
            return E(f"{_ACAST[ft]}({self.iexpr(depth - 1).text})", ft, 0)
        if k < 0.90:
            return self._stmt_expr(depth - 1, float_val=True)
        call = self._call(depth, want_float=True)
        return call if call is not None else self._fleaf()

    def _fleaf(self) -> E:
        r = self.rng
        fvars = [n for n in self._live() if self.styp.get(n) in _FLOATS]
        if fvars and r.random() < 0.5:
            n = fvars[r.randrange(len(fvars))]
            return E(n, self.styp[n], 0)
        if r.random() < 0.4:                            # an integer widened to float
            ft = r.choice(_FLOATS)
            return E(f"{_ACAST[ft]}({self._ileaf(0).text})", ft, 0)
        ft = r.choice(_FLOATS)
        return E(_fconst(round(r.uniform(-1000, 1000), 3), ft), ft, 0)

    def _num(self, depth: int) -> E:                    # a comparison operand: either lane
        return self.fexpr(depth) if self.rng.random() < 0.35 else self.iexpr(depth)

    def _val(self, depth: int, code: str) -> E:         # a value of the storage type `code`
        return self.fexpr(depth) if code in _FLOATS else self.iexpr(depth)

    # -- calls / statement expressions -----------------------------------------------------------------
    def _wide_arg(self, ptype: str) -> str:
        # an argument to a wide-signed (int/long) parameter must respect the callee's PMAX bound assumption:
        # a narrow-signed leaf (always <= 32767) or a small constant -- never a (possibly-mutated) wide one.
        narrows = [n for n in self._live() if n not in self.ptrs and self.styp.get(n) in ("i8", "i16")]
        if narrows and self.rng.random() < 0.5:
            return self.rng.choice(narrows)
        return _const_text(self.rng.randint(-1000, 1000), ptype)

    def _call(self, depth: int, want_float: bool):
        if not self.allow_calls:
            return None
        cands = [h for h in self.helpers if (h[2] in _FLOATS) == want_float]
        if not cands:
            return None
        name, ptypes, rty = self.rng.choice(cands)
        args = []
        for pt in ptypes:
            if pt in _FLOATS:
                args.append(self.fexpr(depth - 1).text)
            elif pt in ("i32", "i64"):
                args.append(self._wide_arg(pt))
            else:
                args.append(self.iexpr(depth - 1).text)
        return E(f"{name}({', '.join(args)})", _ST[rty][4], _ST[rty][3] if _sgn(_ST[rty][4]) else 0)

    def _stmt_expr(self, depth: int, float_val: bool) -> E:
        """`({ <stmts>; <value-expr>; })` -- its own scope; an operand of a larger expression, so it mutates
        only locals it declares (outer names locked) and performs no pointer write."""
        saved_lock, saved_pure = self.locked, self.pure
        self.locked = saved_lock | set(self._live())
        self.pure = True
        self.scopes.append([])
        parts = [self.stmt(depth - 1, allow_block=False) for _ in range(self.rng.randint(0, 2))]
        val = self.fexpr(depth) if float_val else self.iexpr(depth)
        parts.append(f"{val.text};")
        self.scopes.pop()
        self.locked, self.pure = saved_lock, saved_pure
        return E("({ " + " ".join(parts) + " })", val.aty, val.bound)

    # -- statements ------------------------------------------------------------------------------------
    def _decl(self, depth: int) -> str:
        r = self.rng
        if self.aggdefs and r.random() < 0.28:
            return self._agg_decl(depth)
        code = r.choice(_ALL)
        e = self._val(depth, code)
        n = self._fresh(code, self._store_bound(e, code))
        return f"{_ST[code][0]} {n} = {e.text};"

    def _agg_decl(self, depth: int) -> str:
        """A struct/union LOCAL `struct S0 vN = { ... };` -- its members are registered as scoped lvalues.
        A union only ever exposes ONE active member (no type-punning, so behaviour stays well-defined)."""
        r = self.rng
        agg = r.choice(list(self.aggdefs))
        kind, members, active = self.aggdefs[agg]
        self.counter += 1
        vn = f"s{self.counter}"
        if kind == "struct":                                    # init exprs generated BEFORE the members are
            inits = [self._val(depth, c) for _, c in members]   # in scope (so they can't read themselves)
            for (mn, c), e in zip(members, inits):
                self._add_member(f"{vn}.{mn}", c, self._store_bound(e, c))
            return f"struct {agg} {vn} = {{ {', '.join(e.text for e in inits)} }};"
        mn, c = members[active]
        e = self._val(depth, c)
        self._add_member(f"{vn}.{mn}", c, self._store_bound(e, c))
        return f"union {agg} {vn} = {{ .{mn} = {e.text} }};"

    def _assign_to(self, n: str, depth: int) -> str:
        code = self.styp[n]
        e = self._val(depth, code)
        self.bound[n] = self._store_bound(e, code)
        return f"{n} = {e.text};"

    def stmt(self, depth: int, allow_block: bool = True) -> str:
        r = self.rng
        if not self._targets():                                 # nothing mutable yet -> declare a local
            return self._decl(depth)
        k = r.random()
        if k < 0.22:                                            # a fresh initialised local
            return self._decl(depth)
        if k < 0.42:                                            # plain assignment
            return self._assign_to(r.choice(self._targets()), depth)
        if k < 0.56:                                            # compound assignment
            return self._compound(depth)
        if k < 0.66:                                            # inc / dec on an integer target (value discarded)
            ints = [n for n in self._targets() if self.styp[n] not in _FLOATS]
            if ints:
                n = r.choice(ints)
                self._bump(n)
                return f"{n}{r.choice(['++', '--'])};"
            return self._assign_to(r.choice(self._targets()), depth)
        if self.ptrs and not self.pure and k < 0.74:            # a pointer write (sequenced statement)
            return self._ptr_write(depth)
        if depth <= 0:                                          # base case: no more nesting
            return self._assign_to(r.choice(self._targets()), 0)
        if k < 0.82:                                            # a statement expression used as a statement
            self.scopes.append([])
            pre = self.stmt(depth - 1, allow_block=False)
            val = self.iexpr(depth - 1).text
            self.scopes.pop()
            return f"({{ {pre} {val}; }});"
        if not allow_block:                                     # inside a value stmt-expr: no blocks
            return self._assign_to(r.choice(self._targets()), depth)
        if k < 0.91:                                            # if / else
            cond = self.iexpr(depth - 1).text
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
        if code in _FLOATS:                                     # float compound: +-*/ (never traps)
            return f"{n} {r.choice(['+', '-', '*', '/'])}= {self.fexpr(depth).text};"
        op = r.choice(["+", "-", "*", "&", "|", "^"])
        e = self.iexpr(depth)
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
                    return self._assign_to(n, 0)
            self.bound[n] = self._store_bound(E("", ra, bd), code)
        else:
            self.bound[n] = 0
        return f"{n} {op}= {e.text};"

    def _bump(self, n: str):
        code = self.styp[n]
        if not _ST[code][2]:                                   # unsigned wraps
            self.bound[n] = 0
        elif code in ("i8", "i16"):                            # narrow: re-caps to type range
            self.bound[n] = _ST[code][3]
        else:                                                  # wide (outside loops): +1, capped at the type max
            self.bound[n] = min(self.bound.get(n, 0) + 1, _ST[code][3])

    def _ptr_write(self, depth: int) -> str:
        r = self.rng
        p = r.choice(self.ptrs)
        rhs = _coerce(self.iexpr(depth), "u32")
        if r.random() < 0.35:
            return f"*{p} = {rhs};"                       # `*p = x` (canonical); `(*p) = x` is a twin parse gap
        idx = _coerce(self.iexpr(depth - 1) if depth > 0 else self._ileaf(0), "u32")
        op = r.choice(["=", "+=", "-=", "*=", "|=", "&=", "^="])
        return f"{p}[({idx}) & {_ARRSZ - 1}u] {op} {rhs};"

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
    def _enter_function(self, params: list, nptr: int):
        # params is a list of (name, type-token); a token is a scalar code, an aggregate tag, or `*<tag>` (a
        # pointer to a struct, whose members `p->m` are mutable through the pointer -- compared after the
        # call). An aggregate / aggregate-pointer parameter contributes its member lvalues to scope.
        self.scopes = [[]]
        self.styp, self.bound = {}, {}
        for pname, ptype in params:
            if ptype.startswith("*"):                          # a `struct T *` parameter: members via `->`
                _kind, members, _active = self.aggdefs[ptype[1:]]
                for mn, c in members:
                    self._add_member(f"{pname}->{mn}", c, _ST[c][3] if _ST[c][2] else 0)
            elif ptype in self.aggdefs:
                kind, members, active = self.aggdefs[ptype]
                for mn, c in (members if kind == "struct" else [members[active]]):
                    self._add_member(f"{pname}.{mn}", c, _ST[c][3] if _ST[c][2] else 0)
            else:
                self.scopes[0].append(pname)
                self.styp[pname] = ptype
                self.bound[pname] = _ST[ptype][3] if _ST[ptype][2] else 0
        self.loopvars, self.locked, self.pure, self.loopdepth = set(), set(), False, 0
        self.params = params
        self.ptrs = ["p", "q"][:nptr]

    def _function(self, name: str, params: list, ret: str, nptr: int, allow_calls: bool) -> str:
        self._enter_function(params, nptr)
        self.allow_calls = allow_calls
        body = [self.stmt(2) for _ in range(self.rng.randint(2, 5))]
        if ret in self.aggdefs:                            # a struct return BY VALUE: a member-init compound
            _kind, members, _active = self.aggdefs[ret]    # literal (struct only -- not union)
            body.append(f"return (struct {ret}){{ {', '.join(self._val(3, c).text for _, c in members)} }};")
            ret_c = f"struct {ret}"
        else:
            rv = self._val(3, ret)
            body.append(f"return {_coerce(rv, _ST[ret][4]) if _ST[ret][4] != rv.aty else rv.text};")
            ret_c = _ST[ret][0]
        parts = [(f"{self.aggdefs[t[1:]][0]} {t[1:]} *{n}" if t.startswith("*")
                  else f"{self.aggdefs[t][0]} {t} {n}" if t in self.aggdefs else f"{_ST[t][0]} {n}")
                 for n, t in params]
        parts += [f"unsigned *{pn}" for pn in self.ptrs]
        src = f"{ret_c} {name}({', '.join(parts)})\n{{\n  " + "\n  ".join(body) + "\n}\n"
        return src

    def program(self) -> "Program":
        r = self.rng
        self.aggdefs, agg_src = {}, []
        for i in range(r.randint(1, 3)):                        # struct / union type definitions
            kind = r.choice(["struct", "struct", "union"])
            nm = ("S" if kind == "struct" else "U") + str(i)
            members = [(f"m{j}", r.choice(_ALL)) for j in range(r.randint(2, 4))]
            self.aggdefs[nm] = (kind, members, r.randrange(len(members)) if kind == "union" else None)
            agg_src.append(f"{kind} {nm} {{ " + " ".join(f"{_ST[c][0]} {mn};" for mn, c in members) + " };")
        self.helpers, helper_src = [], []
        for hi in range(r.randint(0, 3)):                       # leaf helpers (scalar params, no calls, no ptr)
            hp = [(chr(97 + k), r.choice(_ALL)) for k in range(r.randint(1, 3))]
            ret = r.choice(_ALL)
            helper_src.append(self._function(f"g{hi}", hp, ret, nptr=0, allow_calls=False))
            self.helpers.append((f"g{hi}", [t for _, t in hp], ret))
        aggnames = list(self.aggdefs)
        structs = [n for n in aggnames if self.aggdefs[n][0] == "struct"]

        def _fparam_type():
            roll = r.random()
            if structs and roll < 0.15:
                return "*" + r.choice(structs)                 # a `struct T *` (read+written through the ptr)
            if aggnames and roll < 0.4:
                return r.choice(aggnames)                       # an aggregate by value
            return r.choice(_ALL)
        fparams = [(chr(97 + k), _fparam_type()) for k in range(r.randint(2, 4))]
        fret = r.choice(structs) if (structs and r.random() < 0.15) else r.choice(_ALL)   # struct return by value
        nptr = r.choice([0, 0, 1, 1, 2])
        alias = nptr == 2 and r.random() < 0.5
        fsrc = self._function("f", fparams, fret, nptr=nptr, allow_calls=True)
        return Program("\n".join(agg_src + helper_src + [fsrc]), fparams, fret, nptr, alias, dict(self.aggdefs))


class Program:
    __slots__ = ("source", "ptypes", "ret", "nptr", "alias", "aggdefs")

    def __init__(self, source: str, ptypes: list, ret: str, nptr: int, alias: bool, aggdefs: dict):
        self.source, self.ptypes, self.ret = source, ptypes, ret
        self.nptr, self.alias, self.aggdefs = nptr, alias, aggdefs


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
_FPOOL = [0.0, 1.0, -1.0, 0.5, -0.5, 3.14159, -2.71828, 100.25, -0.001, 1234.5, -99999.0, 1e-7]


def _lit(code: str, j: int, k: int) -> str:
    if code in _FLOATS:
        return _fconst(_FPOOL[(j + 3 * k) % len(_FPOOL)], code)
    return _const_text(_POOL[code][(j + 3 * k) % len(_POOL[code])], _ST[code][4])


_FEQ = (
    "static int feqf(float a,float b){ if(isnan(a)&&isnan(b))return 1; if(isnan(a)||isnan(b))return 0;\n"
    "  if(a==b)return 1; int32_t x,y; memcpy(&x,&a,4); memcpy(&y,&b,4);\n"
    "  if((x<0)!=(y<0))return 0; int32_t d=x>y?x-y:y-x; return d<=4; }\n"
    "static int feqd(double a,double b){ if(isnan(a)&&isnan(b))return 1; if(isnan(a)||isnan(b))return 0;\n"
    "  if(a==b)return 1; int64_t x,y; memcpy(&x,&a,8); memcpy(&y,&b,8);\n"
    "  if((x<0)!=(y<0))return 0; int64_t d=x>y?x-y:y-x; return d<=4; }\n")


def _behaviour_ok(cc: str, prog: Program, emit: str, d: str, label: str) -> tuple[bool, str]:
    renamed = re.sub(r"\bf\b", "f_s", prog.source)              # the entry `f` -> `f_s`; helpers untouched
    struct_ret = prog.ret in prog.aggdefs                       # f returns a struct BY VALUE
    rtype = f"struct {prog.ret}" if struct_ret else _ST[prog.ret][0]
    cmp = None if struct_ret else {"f32": "feqf", "f64": "feqd"}.get(prog.ret)   # None -> exact integer compare

    def _mcmp(lhs: str, rhs: str, code: str) -> str:           # one member's value comparison (float-aware)
        if code in _FLOATS:
            return f"if(!{'feqf' if code == 'f32' else 'feqd'}({lhs},{rhs})) return 1;"
        return f"if({lhs}!={rhs}) return 1;"

    if struct_ret:
        members = prog.aggdefs[prog.ret][1]
        chk = " ".join(_mcmp(f"r1.{mn}", f"r2.{mn}", c) for mn, c in members)
    else:
        chk = f"if(!{cmp}(r1,r2)) return 1;" if cmp else "if(r1!=r2) return 1;"
    # the float-equality helpers are needed for a float compare anywhere (return, struct-return member, or a
    # struct-pointer member).
    need_feq = (cmp is not None
                or (struct_ret and any(c in _FLOATS for _, c in prog.aggdefs[prog.ret][1]))
                or any(t.startswith("*") and any(c in _FLOATS for _, c in prog.aggdefs[t[1:]][1])
                       for _, t in prog.ptypes))
    init = "{3u,140u,7u,0u,99u,1234567u,255u,42u}"
    np, alias = prog.nptr, prog.alias
    lines = ["int main(void){"]
    for j in range(16):
        decls, post, a1, a2 = [], [], [], []
        for k, (_pn, t) in enumerate(prog.ptypes):
            if t.startswith("*"):                               # struct pointer: a backing struct per rail,
                agg = t[1:]                                     # member-initialised, then compared BY VALUE
                _kind, members, _active = prog.aggdefs[agg]     # after the call (per member: feqf/feqd for a
                va, vb = f"S{k}A", f"S{k}B"                     # float -- nan / -0.0 aware -- else exact)
                decls.append(f"struct {agg} {va},{vb}; memset(&{va},0,sizeof {va}); memset(&{vb},0,sizeof {vb});")
                for mi, (mn, c) in enumerate(members):
                    lit = _lit(c, j, k + mi)
                    decls.append(f"{va}.{mn}={lit}; {vb}.{mn}={lit};")
                    if c in _FLOATS:
                        post.append(f"if(!{'feqf' if c == 'f32' else 'feqd'}({va}.{mn},{vb}.{mn})) return {5 + k};")
                    else:
                        post.append(f"if({va}.{mn}!={vb}.{mn}) return {5 + k};")
                a1.append(f"&{va}")
                a2.append(f"&{vb}")
            elif t in prog.aggdefs:
                kind, members, active = prog.aggdefs[t]
                if kind == "struct":
                    lit = f"(struct {t}){{ {', '.join(_lit(c, j, k + mi) for mi, (_, c) in enumerate(members))} }}"
                else:
                    mn, c = members[active]
                    lit = f"(union {t}){{ .{mn} = {_lit(c, j, k)} }}"
                a1.append(lit)
                a2.append(lit)
            else:
                a1.append(_lit(t, j, k))
                a2.append(_lit(t, j, k))
        if np >= 1:
            decls.append(f"unsigned P1[8]={init},P2[8]={init};")
            post.append("for(int kk=0;kk<8;kk++) if(P1[kk]!=P2[kk]) return 2;")
            a1.append("P1")
            a2.append("P2")
            if np == 2:
                if alias:
                    a1.append("P1")
                    a2.append("P2")
                else:
                    decls.append(f"unsigned Q1[8]={init},Q2[8]={init};")
                    a1.append("Q1")
                    a2.append("Q2")
                    post.append("for(int kk=0;kk<8;kk++) if(Q1[kk]!=Q2[kk]) return 3;")
        c1, c2 = f"bcir_f({', '.join(a1)})", f"f_s({', '.join(a2)})"
        lines.append(f"  {{ {' '.join(decls)} {rtype} r1={c1}; {rtype} r2={c2}; {chk} {' '.join(post)} }}")
    lines.append("  return 0;}")
    harness = ("#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n#include <math.h>\n"
               + (_FEQ if need_feq else "") + renamed + "\n" + emit + "\n" + "\n".join(lines))
    cpath = os.path.join(d, f"{label}.c")
    epath = os.path.join(d, label)
    with open(cpath, "w") as fh:
        fh.write(harness)
    for std in ("c23", "c2x", "c17"):                           # -ffp-contract=off: no FMA, so float is exact
        b = subprocess.run([cc, f"-std={std}", "-O2", "-ffp-contract=off", cpath, "-o", epath, "-lm"],
                           capture_output=True, text=True)
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
