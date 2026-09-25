"""The G10 fixtures and grader (S5-B): escape analysis, indirect-call narrowing and the effect
footprint of a lowered C unit (`bcir/frontends/cfront/escape.py`; the C twin is
`bcir_cfront_escape` / `bcir_cfront_effects`, `bcir-cc --emit-escape` / `--emit-effects`).

One module the tests (`test_escape_analysis.py`, `test_c_cfront.py`), the gate
(`tools/perf/check_escape.py`) and the harness (`tools/perf/gemplus_baseline.py --group escape`)
share. Its judges share no code with the analysis:

  * `generate(seed)` writes a C unit whose escape verdicts and indirect-call target sets are KNOWN
    BY CONSTRUCTION (a local array read in place, lent to a reader, captured into a global; a
    function pointer assigned one or two known functions), with heap buffers, a pointer made from an
    integer, a static writer and a callback in a file-scope ops table mixed in, and two exported
    functions that differ only through a static local of a helper both call;
  * `witness(...)` runs each pair of functions in both orders, in separate processes, over shared
    buffers (one per pointee type, so two pointer parameters alias) and hashes every result, buffer
    and global: two different hashes are a definite witness that the pair does NOT commute;
  * `corpus()` is the cfront corpus (`runtime/c/cfront_*.c`) the oracle compiles, a checkout asset;
  * `form_units()` holds every declaration kind (scalar, pointer, array, struct, pointer to struct,
    array of pointers, pointer to pointer, and the volatile ones) against every access form (value,
    index, dereference, member, address taken, passed to a reader, a writer and an unknown callee),
    in every storage place (file scope, static, automatic, parameter) -- the constructs a corpus
    written for other purposes happens to lack.

The rows (`measure`):

    escape.unproved           declared-extent local arrays of the cfront corpus (the roadmap's
                              escape candidates) NOT proved to stay in their activation: the
                              candidates less the ones proved nonescaping
    icall.unknown             indirect call sites of the cfront corpus not narrowed to known
                              functions of the unit: the sites less the known ones
    icall.unresolved          indirect call sites of the cfront corpus not resolved to exactly one
                              function: the sites less the resolved ones
    escape.verdict.mismatch   named locals of the generated units whose verdict is not the one
                              they have by construction (none reported counts)
    icall.target.mismatch     indirect call sites of the generated units whose narrowed target set
                              is not the one they have by construction (not narrowed counts)
    effects.commute.unsound   function pairs of the corpus and the generated units the footprints
                              call commuting while the witness shows the two orders diverge
                              (needs a C compiler)
    effects.parity.mismatch   units of the corpus, the generated set and the forms whose `bcir-cc
                              --emit-effects` differs from the oracle's report; a unit the twin
                              refuses counts, unless it is one of the pinned preprocessor limits
    escape.parity.mismatch    the same for `bcir-cc --emit-escape` (both need a C compiler and
                              the checkout's runtime/c)

A row that needs a tool the host lacks is left OUT of the result, never zero by default (L2).
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import tempfile

#: Every row `measure` reports when every tool is present; the C-compiler rows are CC_ROWS.
ROWS = (
    "escape.unproved",
    "icall.unknown",
    "icall.unresolved",
    "escape.verdict.mismatch",
    "icall.target.mismatch",
    "effects.commute.unsound",
    "effects.parity.mismatch",
    "escape.parity.mismatch",
)
CC_ROWS = ("effects.commute.unsound", "effects.parity.mismatch", "escape.parity.mismatch")
#: The generated units the rows grade (fixed: the rows are exact).
SEEDS = tuple(range(24))
#: The corpus units the twin's preprocessor does not take, so only the oracle reports on them. The
#: set is exact: any other unit the twin refuses is a parity failure, and the parity test fails when
#: one of these starts to compile (the pin is then stale).
TWIN_PREPROCESSOR_LIMITS = frozenset({"cfront_sec_cppmacro.c"})

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS_DIR = os.path.join(_ROOT, "runtime", "c")
#: The witness build: no optimizer, fixed addresses (no PIE, so a function or global address hashes
#: the same in both processes), automatic storage zeroed (an uninitialized read would otherwise
#: differ between the two processes and forge a divergence).
WITNESS_FLAGS = ("-std=gnu11", "-O0", "-w", "-fno-pie", "-no-pie", "-ftrivial-auto-var-init=zero")


def find_cc() -> str | None:
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


# --- the generated units: escape verdicts and indirect-call targets known by construction --------


def generate(seed: int) -> tuple[str, dict, dict]:
    """A C unit for `seed` and its truths: `{(function, local): verdict}` and
    `{function: (target, ...)}` (one indirect site per function at most). Every function the
    witness drives takes unsigned scalars and `unsigned *` (so it can bind shared buffers); the
    static helpers take pointers only from inside the unit, except `cb_w`, which the file-scope ops
    table hands to callers the unit cannot see."""
    rnd = random.Random(seed)
    lines = [
        "#include <stdint.h>",
        "#include <stdlib.h>",
        "unsigned gs0;",
        "unsigned gs1;",
        "unsigned ga0[8];",
        "unsigned ga1[8];",
        "unsigned *gp;",
        "struct ops { unsigned (*fn)(unsigned *, unsigned); };",
        "static unsigned h_pure(unsigned x) { return x * 3u + 1u; }",
        "static unsigned h_twice(unsigned x) { return x + x; }",
        "static unsigned h_read(unsigned *p, unsigned i) { return p[i & 7u]; }",
        "static unsigned h_capture(unsigned *p, unsigned i) { gp = p; return i; }",
        "static unsigned h_write(unsigned *p, unsigned i) { p[i & 7u] = i; return i; }",
        "static unsigned cb_w(unsigned *p, unsigned i) { p[i & 7u] = i + 1u; return i; }",
        "struct ops tab = { cb_w };",
    ]
    truth_escape: dict = {}
    truth_icall: dict = {}
    fns: list = []
    for k in range(rnd.randint(3, 6)):
        name = f"f{k}"
        static = rnd.random() < 0.3
        has_ptr = rnd.random() < 0.5
        params = ["unsigned x"] + (["unsigned *p"] if has_ptr else [])
        body = ["  unsigned v = x;"]
        for _ in range(rnd.randint(2, 5)):
            op = rnd.choice(
                (
                    "gsr", "gsw", "gar", "gaw", "pr", "pw", "stat", "loc", "lend", "esc", "call",
                    "icall", "heap", "hwrite", "forge", "cb",
                )
            )  # fmt: skip
            if op == "gsr":
                body.append(f"  v = v + gs{rnd.randint(0, 1)};")
            elif op == "gsw":
                body.append(f"  gs{rnd.randint(0, 1)} = v;")
            elif op == "gar":
                body.append(f"  v = v + ga{rnd.randint(0, 1)}[v & 7u];")
            elif op == "gaw":
                body.append(f"  ga{rnd.randint(0, 1)}[x & 7u] = v;")
            elif op == "pr" and has_ptr:
                body.append("  v = v + p[x & 7u];")
            elif op == "pw" and has_ptr:
                body.append("  p[v & 7u] = x;")
            elif op == "stat":
                sv = f"st{len(body)}"
                body.append(f"  static unsigned {sv}; {sv} = {sv} + 1u; v = v + {sv};")
            elif op == "loc":  # read and written in place: private to the activation
                t = f"t{len(body)}"
                body.append(f"  unsigned {t}[8]; {t}[x & 7u] = v; v = v + {t}[(x + 1u) & 7u];")
                truth_escape[(name, t)] = "nonescaping"
            elif op == "lend":  # handed to a reader that keeps nothing: crosses a call
                t = f"t{len(body)}"
                body.append(f"  unsigned {t}[8]; {t}[x & 7u] = v; v = v + h_read({t}, x);")
                truth_escape[(name, t)] = "lent"
            elif op == "esc":  # its address is stored in a global: reachable from outside
                t = f"t{len(body)}"
                body.append(f"  unsigned {t}[8]; {t}[x & 7u] = v; v = v + h_capture({t}, x);")
                truth_escape[(name, t)] = "escaping"
            elif op == "call" and fns:
                cn, cptr = rnd.choice(fns)
                args = "v" + (", p" if (cptr and has_ptr) else (", ga0" if cptr else ""))
                body.append(f"  v = v + {cn}({args});")
            elif op == "icall" and name not in truth_icall:
                targets = sorted(rnd.sample(["h_pure", "h_twice"], rnd.randint(1, 2)))
                body.append(f"  unsigned (*fp)(unsigned) = {targets[0]};")
                if len(targets) > 1:
                    body.append(f"  if (x & 1u) fp = {targets[1]};")
                body.append("  v = v + fp(v);")
                truth_icall[name] = tuple(targets)
            elif op == "heap":  # a heap buffer private to the activation, lent to a writer
                hb = f"hb{len(body)}"
                body.append(
                    f"  unsigned *{hb} = calloc(8u, sizeof(unsigned)); v = v + h_write({hb}, x);"
                    f" v = v + {hb}[x & 7u]; free({hb});"
                )
            elif op == "hwrite":
                tgt = "p" if (has_ptr and rnd.random() < 0.5) else f"ga{rnd.randint(0, 1)}"
                body.append(f"  v = v + h_write({tgt}, v);")
            elif op == "forge":  # a pointer made from an integer (a round trip through uintptr_t)
                g = rnd.randint(0, 1)
                body.append(f"  {{ unsigned *q = (unsigned *)(uintptr_t)ga{g}; q[x & 7u] = v; }}")
            elif op == "cb":
                tgt = "p" if (has_ptr and rnd.random() < 0.5) else f"ga{rnd.randint(0, 1)}"
                body.append(f"  v = v + cb_w({tgt}, x);")
        body.append("  return v;")
        lines.append(f"{'static ' if static else ''}unsigned {name}({', '.join(params)}) {{")
        lines += body
        lines.append("}")
        fns.append((name, has_ptr))
    # two exported functions that differ only through a static local of a helper they both call:
    # the one pair whose orders only that static tells apart. Fixed text after the random functions,
    # so the random stream -- and the truths -- are what they were.
    lines += [
        "static unsigned tick(void) { static unsigned n; n = n + 1u; return n; }",
        "unsigned s_a(unsigned x) { return x + tick(); }",
        "unsigned s_b(unsigned x) { return x * 2u + tick(); }",
    ]
    return "\n".join(lines) + "\n", truth_escape, truth_icall


# --- the dynamic commute witness ----------------------------------------------------------------


def _param_ok(ct) -> bool:
    if ct.kind == "scalar":
        return True
    if ct.kind in ("pointer", "array") and ct.of is not None:
        if ct.of.kind == "scalar":
            return True
        if ct.of.kind in ("struct", "union"):  # a pointee of scalar fields: any bytes are valid
            return all(ft.kind == "scalar" for _n, ft, *_ in ct.of.fields)
    return False


def callable_fn(lf, init_refs=frozenset()) -> bool:
    """Whether the witness may drive `lf` directly: scalar and scalar-pointee arguments, a scalar
    or pointer result, not variadic. A static function is called only from inside its unit, so one
    with pointer parameters is driven only when a file-scope initializer hands it out."""
    if getattr(lf, "variadic", False):
        return False
    if (
        getattr(lf, "static_fn", False)
        and lf.name not in init_refs
        and any(ct.kind != "scalar" for _n, _r, ct in lf.params)
    ):
        return False
    if lf.ret_type.kind not in ("scalar", "void", "pointer"):
        return False
    return all(_param_ok(ct) for _n, _r, ct in lf.params)


def _elem_key(ct) -> str:
    from bcir.frontends.cfront.emit import _cname  # noqa: PLC0415

    return _cname(ct.of) if ct.of is not None else "uint32_t"


def witness_program(source: str, lowered, pairs) -> str:
    """One C program driving every pair: `main(pair, order)` runs `a` then `b` (order 0) or `b`
    then `a` (order 1) over seeded trials on shared buffers and prints one hash."""
    from bcir.frontends.cfront.emit import _cname  # noqa: PLC0415

    fns = lowered.functions
    pools: dict = {}  # element type -> (pool name, whether it hashes by value)
    for a, b in pairs:
        for fn in (a, b):
            for _n, _r, ct in fns[fn].params:
                if ct.kind in ("pointer", "array"):
                    by_value = ct.of is not None and ct.of.is_integer
                    pools.setdefault(_elem_key(ct), (f"pool{len(pools)}", by_value))
    # the globals the unit defines, hashed -- a pointer's own value is an address (a dangling
    # local, a heap block) that differs between processes; what it points to is hashed through
    # the pools and the globals
    # An integer is hashed by VALUE: the padding bits of a `_BitInt(N)` (or of any integer
    # narrower than its storage) are unspecified, so its bytes can differ between two runs that
    # computed the same value.
    gnames = [
        (g[0], g[1].is_integer)
        for g in getattr(lowered, "globals_decl", ())
        if not g[3] and getattr(g[1], "kind", "") not in ("pointer", "funcptr")
    ]

    def call(fn, tag):
        """(argument setup, the call, the mix of its result): the arguments are drawn BEFORE
        either call, in a fixed order, and the results mixed after both, in a fixed order."""
        lf = fns[fn]
        setup, args = [], []
        for i, (_n, _r, ct) in enumerate(lf.params):
            if ct.kind in ("pointer", "array"):
                args.append(pools[_elem_key(ct)][0])
            else:
                setup.append(f"{_cname(ct)} _a{tag}{i} = ({_cname(ct)})(rng() % 48u);")
                args.append(f"_a{tag}{i}")
        c = f"{fn}({', '.join(args)})"
        if lf.ret_type.kind == "void":
            return " ".join(setup), f"{c};", ""
        if lf.ret_type.kind == "pointer":  # an address: only whether it is null is portable
            return " ".join(setup), f"_r{tag} = {c} != 0;", f"_mix(&_r{tag}, sizeof _r{tag});"
        if lf.ret_type.is_integer:
            return (
                " ".join(setup),
                f"_r{tag} = (uint64_t)({c});",
                f"_mix(&_r{tag}, sizeof _r{tag});",
            )
        return (
            " ".join(setup),
            f"{{ {_cname(lf.ret_type)} _v = {c}; _r{tag} = 0;"
            f" memcpy(&_r{tag}, &_v, sizeof _v < 8 ? sizeof _v : 8); }}",
            f"_mix(&_r{tag}, sizeof _r{tag});",
        )

    cases = []
    for idx, (a, b) in enumerate(pairs):
        sa, ca, ma = call(a, "a")
        sb, cb, mb = call(b, "b")
        cases.append(
            f"    case {idx}: for (int t = 0; t < 24; t++) {{ _s = 0x9E3779B97F4A7C15u + (uint64_t)t;"
            f" _fill(); uint64_t _ra = 0, _rb = 0; {sa} {sb}"
            f" if (order == 0) {{ {ca} {cb} }} else {{ {cb} {ca} }} {ma} {mb} }} break;"
        )
    fill = "".join(
        f" for (unsigned long i = 0; i < sizeof {p}; i++) ((unsigned char *){p})[i] ="
        " (unsigned char)(rng() % 7u);"
        for p, _v in pools.values()
    )
    dump = "".join(
        f"  {{ uint64_t _g = (uint64_t)({g}); _mix(&_g, sizeof _g); }}\n"
        if by_value
        else f"  _mix(&{g}, sizeof {g});\n"
        for g, by_value in gnames
    )
    dump += "".join(
        f"  for (unsigned long i = 0; i < sizeof {p} / sizeof {p}[0]; i++)"
        f" {{ uint64_t _e = (uint64_t){p}[i]; _mix(&_e, sizeof _e); }}\n"
        if by_value
        else f"  _mix({p}, sizeof {p});\n"
        for p, by_value in pools.values()
    )
    decls = "".join(f"static {k} {p}[64];\n" for k, (p, _v) in pools.items())
    return (
        "#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n"
        "#include <stdatomic.h>\n#include <math.h>\n"
        f"{source}\n{decls}"
        "static uint64_t _h = 1469598103934665603u;\n"
        "static void _mix(const void *p, unsigned long n) { const unsigned char *c = p;"
        " for (unsigned long i = 0; i < n; i++) { _h ^= c[i]; _h *= 1099511628211u; } }\n"
        "static uint64_t _s;\n"
        "static uint32_t rng(void) { _s = _s * 6364136223846793005u + 1442695040888963407u;"
        " return (uint32_t)(_s >> 32); }\n"
        f"static void _fill(void) {{ uint64_t save = _s; _s = 12345u;{fill} _s = save; }}\n"
        "int main(int argc, char **argv) {\n"
        "  if (argc < 3) return 2;\n"
        "  int pair = atoi(argv[1]), order = atoi(argv[2]);\n"
        "  switch (pair) {\n" + "\n".join(cases) + "\n    default: return 3;\n  }\n"
        f"{dump}"
        '  printf("%016llx\\n", (unsigned long long)_h);\n'
        "  return 0;\n}\n"
    )


def initializer_names(unit) -> frozenset:
    """Every identifier a file-scope initializer of the parsed `unit` names -- read from the AST,
    not from the lowering the analysis uses, so the judge does not share the fact it judges."""
    import dataclasses  # noqa: PLC0415

    from bcir.frontends.cfront import cast  # noqa: PLC0415

    out: set = set()
    stack = [g.init for g in getattr(unit, "globals", ())]
    while stack:
        n = stack.pop()
        if isinstance(n, cast.Name):
            out.add(n.ident)
        elif isinstance(n, (tuple, list)):
            stack.extend(n)
        elif dataclasses.is_dataclass(n) and not isinstance(n, type):
            stack.extend(getattr(n, f.name) for f in dataclasses.fields(n))
    return frozenset(out)


def witness(res, pairs, cc: str) -> dict:
    """`{(a, b): verdict}` for each pair of the compiled unit `res`: `conflict` (the two orders
    diverge), `agree`, or `skip:<why>` (a signature the harness cannot drive, a unit defining
    `main`, a build or run that failed)."""
    source, lowered = res.source, res.lowered
    refs = initializer_names(res.unit)
    fns = lowered.functions
    ok = [(a, b) for a, b in pairs if callable_fn(fns[a], refs) and callable_fn(fns[b], refs)]
    out = {p: "skip:signature" for p in pairs if p not in ok}
    if not ok:
        return out
    if "main" in fns:
        out.update({p: "skip:main" for p in ok})
        return out
    with tempfile.TemporaryDirectory() as d:
        src, exe = os.path.join(d, "w.c"), os.path.join(d, "w")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write(witness_program(source, lowered, ok))
        try:
            b = subprocess.run(
                [cc, *WITNESS_FLAGS, src, "-o", exe, "-lm"],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired):
            b = None
        if b is None or b.returncode != 0:
            out.update({p: "skip:build" for p in ok})
            return out
        for i, p in enumerate(ok):
            hashes = []
            for order in (0, 1):
                try:
                    r = subprocess.run(
                        [exe, str(i), str(order)], capture_output=True, text=True, timeout=10
                    )
                    hashes.append(r.stdout.strip() if r.returncode == 0 else None)
                except (OSError, subprocess.TimeoutExpired):
                    hashes.append(None)
            if None in hashes:
                out[p] = "skip:run"
            else:
                out[p] = "agree" if hashes[0] == hashes[1] else "conflict"
    return out


def witness_available(cc: str | None) -> bool:
    """Whether `cc` builds the witness (its flags need clang >= 8 or gcc >= 12)."""
    if not cc:
        return False
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "p.c")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write("int main(void) { int x; return x & 0; }\n")
        try:
            b = subprocess.run(
                [cc, *WITNESS_FLAGS, src, "-o", os.path.join(d, "p")],
                capture_output=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return b.returncode == 0


# --- the forms: every declaration kind against every access form, in every storage place ----------

#: What every form unit declares first: a struct with a scalar, a pointer and an array member, the
#: globals a form can leak to or read, an extern the unit cannot see, and two static callees that
#: keep nothing (a reader and a writer).
_FORM_PRELUDE = """#include <stdint.h>
struct S { unsigned a; unsigned *p; unsigned arr[4]; };
unsigned gsink_v;
unsigned *gsink_p;
unsigned gbuf[8];
extern unsigned ext(unsigned *q);
static unsigned keep(unsigned *q, unsigned i) { return q[i & 3u]; }
static unsigned put(unsigned *q, unsigned i) { q[i & 3u] = i; return i; }
"""
#: A declaration kind: its declarator, the variable named `{n}`.
FORM_KINDS = {
    "scalar": "unsigned {n}",
    "ptr": "unsigned *{n}",
    "arr": "unsigned {n}[8]",
    "arr1": "unsigned {n}[1]",
    "st": "struct S {n}",
    "pst": "struct S *{n}",
    "aptr": "unsigned *{n}[4]",
    "pp": "unsigned **{n}",
    "vscalar": "volatile unsigned {n}",
    "vptr": "volatile unsigned *{n}",
    "varr": "volatile unsigned {n}[8]",
    "vst": "volatile struct S {n}",
    "vpst": "volatile struct S *{n}",
}
#: An access form: one statement over the variable `{x}`, the running value `v` and the index `i`.
FORM_ACCESSES = {
    "val": "v = v + (unsigned)(uintptr_t){x};",
    "idx_ld": "v = v + {x}[i & 3u];",
    "idx_st": "{x}[i & 3u] = v;",
    "deref_ld": "v = v + *{x};",
    "deref_st": "*{x} = v;",
    "mem_ld": "v = v + {x}.a;",
    "mem_st": "{x}.a = v;",
    "arrow_ld": "v = v + {x}->a;",
    "arrow_st": "{x}->a = v;",
    "arrow_p_ld": "v = v + {x}->p[i & 3u];",
    "arrow_arr_st": "{x}->arr[i & 3u] = v;",
    "mem_arr_st": "{x}.arr[i & 3u] = v;",
    "addr_glob": "gsink_p = (unsigned *)&{x};",
    "addr_ext": "v = v + ext((unsigned *)&{x});",
    "pass_keep": "v = v + keep((unsigned *){x}, i);",
    "pass_put": "v = v + put((unsigned *){x}, i);",
    "pass_ext": "v = v + ext((unsigned *){x});",
    "assign_glob": "{x} = gbuf;",
    "pp_ld": "v = v + **{x};",
    "pp_idx": "v = v + {x}[0][i & 3u];",
    "aptr_set": "{x}[i & 3u] = gbuf;",
}
#: The forms graded, per storage place and kind: every (kind, access) whose function is C that clang
#: accepts and that both rails lower -- chosen by sweeping all of them, which is how the twin's
#: file-scope pointer (touched in place) and its index load through a volatile pointer (an ordinary
#: load) were found. Pinned, not rediscovered: a frontend that stops lowering one fails its place's
#: unit, and the parity rows count that. The units are compared, never run.
FORMS = {
    "global": {
        "scalar": "addr_ext addr_glob pass_ext pass_keep pass_put val",
        "ptr": "addr_ext addr_glob assign_glob idx_ld idx_st pass_ext pass_keep pass_put val",
        "arr": "addr_ext addr_glob deref_ld deref_st idx_ld idx_st pass_ext pass_keep pass_put val",
        "arr1": "addr_ext addr_glob idx_ld idx_st pass_ext pass_keep pass_put val",
        "st": "addr_ext addr_glob",
        "pst": "addr_ext addr_glob pass_ext pass_keep pass_put val",
        "aptr": "addr_ext addr_glob aptr_set pass_ext pass_keep pass_put pp_idx val",
        "pp": "addr_ext addr_glob aptr_set pass_ext pass_keep pass_put pp_idx val",
        "vscalar": "addr_ext addr_glob pass_ext pass_keep pass_put val",
        "vptr": "addr_ext addr_glob assign_glob idx_ld pass_ext pass_keep pass_put val",
        "varr": "addr_ext addr_glob deref_ld deref_st idx_ld pass_ext pass_keep pass_put val",
        "vst": "addr_ext addr_glob",
        "vpst": "addr_ext addr_glob pass_ext pass_keep pass_put val",
    },
    "static": {
        "scalar": "addr_ext addr_glob pass_ext pass_keep pass_put val",
        "ptr": "addr_ext addr_glob assign_glob deref_ld deref_st idx_ld idx_st pass_ext pass_keep pass_put val",
        "arr": "addr_ext addr_glob deref_st idx_ld idx_st pass_ext pass_keep pass_put val",
        "arr1": "addr_ext addr_glob deref_st idx_ld idx_st pass_ext pass_keep pass_put val",
        "st": "addr_ext addr_glob mem_arr_st mem_ld mem_st",
        "pst": "addr_ext addr_glob arrow_arr_st arrow_ld arrow_p_ld arrow_st pass_ext pass_keep pass_put val",
        "aptr": "addr_ext addr_glob aptr_set pass_ext pass_keep pass_put pp_idx val",
        "pp": "addr_ext addr_glob aptr_set pass_ext pass_keep pass_put pp_idx pp_ld val",
        "vscalar": "addr_ext addr_glob pass_ext pass_keep pass_put val",
        "vptr": "addr_ext addr_glob deref_ld deref_st idx_ld idx_st pass_ext pass_keep pass_put val",
        "varr": "addr_ext addr_glob deref_ld deref_st idx_ld idx_st pass_ext pass_keep pass_put val",
        "vst": "addr_ext addr_glob mem_arr_st mem_ld mem_st",
        "vpst": "addr_ext addr_glob arrow_arr_st arrow_ld arrow_p_ld arrow_st pass_ext pass_keep pass_put val",
    },
    "local": {
        "scalar": "addr_ext addr_glob pass_ext pass_keep pass_put val",
        "ptr": "addr_ext addr_glob assign_glob deref_ld deref_st idx_ld idx_st pass_ext pass_keep pass_put val",
        "arr": "addr_ext addr_glob deref_st idx_ld idx_st pass_ext pass_keep pass_put val",
        "arr1": "addr_ext addr_glob deref_st idx_ld idx_st pass_ext pass_keep pass_put val",
        "st": "addr_ext addr_glob mem_arr_st mem_ld mem_st",
        "pst": "addr_ext addr_glob arrow_arr_st arrow_ld arrow_p_ld arrow_st pass_ext pass_keep pass_put val",
        "aptr": "addr_ext addr_glob aptr_set pass_ext pass_keep pass_put pp_idx val",
        "pp": "addr_ext addr_glob aptr_set pass_ext pass_keep pass_put pp_idx pp_ld val",
        "vscalar": "addr_ext addr_glob pass_ext pass_keep pass_put val",
        "vptr": "addr_ext addr_glob deref_ld deref_st idx_ld idx_st pass_ext pass_keep pass_put val",
        "varr": "addr_ext addr_glob deref_ld deref_st idx_ld idx_st pass_ext pass_keep pass_put val",
        "vst": "addr_ext addr_glob mem_arr_st mem_ld mem_st",
        "vpst": "addr_ext addr_glob arrow_arr_st arrow_ld arrow_p_ld arrow_st pass_ext pass_keep pass_put val",
    },
    "param": {
        "scalar": "addr_ext addr_glob pass_ext pass_keep pass_put val",
        "ptr": "addr_ext addr_glob assign_glob deref_ld deref_st idx_ld idx_st pass_ext pass_keep pass_put val",
        "st": "addr_ext addr_glob mem_arr_st mem_ld mem_st",
        "pst": "addr_ext addr_glob arrow_arr_st arrow_ld arrow_p_ld arrow_st pass_ext pass_keep pass_put val",
        "pp": "addr_ext addr_glob aptr_set pass_ext pass_keep pass_put pp_idx pp_ld val",
        "vscalar": "addr_ext addr_glob pass_ext pass_keep pass_put val",
        "vptr": "addr_ext addr_glob deref_ld deref_st idx_ld idx_st pass_ext pass_keep pass_put val",
        "varr": "addr_ext addr_glob deref_ld deref_st idx_ld idx_st pass_ext pass_keep pass_put val",
        "vst": "addr_ext addr_glob mem_arr_st mem_ld mem_st",
        "vpst": "addr_ext addr_glob arrow_arr_st arrow_ld arrow_p_ld arrow_st pass_ext pass_keep pass_put val",
    },
}


def form_units() -> list[tuple[str, str]]:
    """`[(place, source)]`: one unit per storage place, one function `f_<kind>_<access>` per form
    (at file scope each has its own global `X_<kind>_<access>`)."""
    out = []
    for place, kinds in FORMS.items():
        lines = [_FORM_PRELUDE]
        for kind, accesses in kinds.items():
            for access in accesses.split():
                fn = f"f_{kind}_{access}"
                var = f"X_{kind}_{access}" if place == "global" else "X"
                decl = FORM_KINDS[kind].format(n=var)
                body = FORM_ACCESSES[access].format(x=var)
                if place == "global":
                    lines.append(f"{decl};")
                    lines.append(
                        f"unsigned {fn}(unsigned i) {{ unsigned v = i; {body} return v; }}"
                    )
                elif place == "param":
                    lines.append(
                        f"unsigned {fn}({decl}, unsigned i) {{ unsigned v = i; {body} return v; }}"
                    )
                else:
                    storage = "static " if place == "static" else ""
                    lines.append(
                        f"unsigned {fn}(unsigned i) {{ {storage}{decl}; unsigned v = i; {body} return v; }}"
                    )
        out.append((place, "\n".join(lines) + "\n"))
    return out


# --- the corpus and the rail parity --------------------------------------------------------------


def corpus() -> list:
    """`[(name, path, source, CompileResult)]` for every `runtime/c/cfront_*.c` the oracle
    compiles, in name order; empty outside a checkout (the wheel does not ship runtime/c)."""
    from bcir.frontends.cfront import compile_unit  # noqa: PLC0415

    if not os.path.isdir(CORPUS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(CORPUS_DIR)):
        if not (name.startswith("cfront_") and name.endswith(".c")):
            continue
        path = os.path.join(CORPUS_DIR, name)
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
        try:
            res = compile_unit(source, check_clang=False, search_paths=[CORPUS_DIR])
        except Exception:  # noqa: BLE001 -- a unit outside the oracle's subset is not in the corpus
            continue
        out.append((name, path, source, res))
    return out


def rail_parity(cc_bin: str, units, may_refuse=frozenset()) -> tuple[int, int, int]:
    """(effects mismatches, escape mismatches, units compared): `bcir-cc --emit-effects` and
    `--emit-escape` against the oracle's reports, byte for byte, over `units` [(path, result)].
    A unit whose file name is in `may_refuse` (a pinned preprocessor limit) and that the twin
    refuses is not compared; any other refusal is a mismatch, so a twin that reports on nothing
    cannot pass."""
    from bcir.frontends.cfront.escape import effects_report, escape_report  # noqa: PLC0415

    eff = esc = compared = 0
    for path, res in units:
        outs = []
        for flag in ("--emit-effects", "--emit-escape"):
            p = subprocess.run([cc_bin, flag, path], capture_output=True, text=True, timeout=60)
            outs.append(p.stdout if p.returncode == 0 else None)
        if outs[0] is None and outs[1] is None and os.path.basename(path) in may_refuse:
            continue
        compared += 1
        eff += outs[0] != effects_report(res.lowered, res.escape)
        esc += outs[1] != escape_report(res.lowered, res.escape)
    return eff, esc, compared


def build_bcir_cc(workdir: str) -> str:
    """The twin's driver, built from the one source list the tests build it with."""
    from bcir.tests.test_c_cfront import _build_bcir_cc  # noqa: PLC0415

    return _build_bcir_cc(workdir)


# --- the rows ------------------------------------------------------------------------------------


def truth_mismatches(seeds=SEEDS) -> tuple[int, int, int, int]:
    """(escape mismatches, named locals, icall mismatches, sites) over the generated units."""
    from bcir.frontends.cfront import compile_unit  # noqa: PLC0415

    esc_bad = esc_n = icall_bad = icall_n = 0
    for seed in seeds:
        source, truth_escape, truth_icall = generate(seed)
        res = compile_unit(source, check_clang=False)
        for (fn, local), want in truth_escape.items():
            esc_n += 1
            esc_bad += res.escape.objects.get(fn, {}).get(local) != want
        for site in res.escape.sites:
            icall_n += 1
            icall_bad += site.targets != truth_icall.get(site.function)
    return esc_bad, esc_n, icall_bad, icall_n


def commute_unsound(results, cc: str) -> tuple[int, int]:
    """(pairs reported commuting that the witness shows diverge, pairs the witness decided) over
    the compiled units `results` (the witness builds each one's preprocessed source)."""
    bad = decided = 0
    for res in results:
        fns = list(res.lowered.functions)
        pairs = [(a, b) for i, a in enumerate(fns) for b in fns[i + 1 :]]
        if not pairs:
            continue
        for (a, b), verdict in witness(res, pairs, cc).items():
            if verdict in ("agree", "conflict"):
                decided += 1
                bad += verdict == "conflict" and res.commute(a, b)
    return bad, decided


def measure(cc: bool = True, bcir_cc: str | None = None) -> dict[str, float]:
    """The rows (module docstring). `cc=False` measures only the rows the interpreter decides;
    with it, the C-compiler rows are left out when no compiler builds the witness, and the parity
    rows when the checkout's runtime/c is absent (never zero by default)."""
    from bcir.frontends.cfront import compile_unit  # noqa: PLC0415

    units = corpus()
    out: dict[str, float] = {}
    if units:
        counts = [res.escape.counts() for _n, _p, _s, res in units]
        out["escape.unproved"] = float(sum(c["candidates"] - c["nonescaping"] for c in counts))
        out["icall.unknown"] = float(sum(c["indirect"] - c["known"] for c in counts))
        out["icall.unresolved"] = float(sum(c["indirect"] - c["resolved"] for c in counts))
    esc_bad, _n, icall_bad, _m = truth_mismatches()
    out["escape.verdict.mismatch"] = float(esc_bad)
    out["icall.target.mismatch"] = float(icall_bad)
    compiler = find_cc() if cc else None
    if compiler and witness_available(compiler):
        generated = [compile_unit(generate(seed)[0], check_clang=False) for seed in SEEDS]
        bad, _decided = commute_unsound([r for _n, _p, _s, r in units] + generated, compiler)
        out["effects.commute.unsound"] = float(bad)
    if compiler and units:
        with tempfile.TemporaryDirectory() as d:
            exe = bcir_cc or build_bcir_cc(d)
            pairs = [(p, r) for _n, p, _s, r in units]
            # the generated units and the forms carry what the corpus lacks: heap, forged
            # pointers, an ops table, every declaration kind against every access form
            extra = [(f"generated_{seed}.c", generate(seed)[0]) for seed in SEEDS]
            extra += [(f"forms_{place}.c", source) for place, source in form_units()]
            for name, source in extra:
                path = os.path.join(d, name)
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(source)
                pairs.append((path, compile_unit(source, check_clang=False)))
            eff, esc, _compared = rail_parity(exe, pairs, TWIN_PREPROCESSOR_LIMITS)
        out["effects.parity.mismatch"] = float(eff)
        out["escape.parity.mismatch"] = float(esc)
    return out
