"""The volatile forms of the C frontend and their grader: every place a C program puts `volatile`,
against every access form, at every element width -- lowered by both cfront rails (the oracle,
`bcir/frontends/cfront/`, and the C twin, `runtime/c/bcir_cfront.c` behind `bcir-cc`) and judged
by Clang, not by either rail.

A `volatile` access is a device access. Both rails must mark it (claim domain MMIO, lane H, hazard
`barriered`, the claim's volatile bit), both must emit C that performs it through a
volatile-qualified lvalue of exactly the accessed type -- a `volatile uint8_t *` store writes one
byte at `p + i`, never four at `p + 4*i` -- and the escape analysis (G10) must see it as reading
and writing unknown memory, so no two of them ever commute. None of that was held: the corpus had
only 32-bit register maps, and both rails had quietly assumed 32-bit registers.

The judges share no code with either frontend:

  * the ORIGINAL program and each rail's emitted C are compiled together by Clang at -O2 with
    inlining off, and the ordered list of volatile loads and stores (with their LLVM types) in each
    original function must equal the list in its emitted `bcir_` twin -- volatility and width
    checked by the compiler that will build the output;
  * each entry runs, original then emitted, from the same seeded memory, globals reset in between,
    and must return the same value and leave the same bytes in every buffer and global it touches;
  * the rails' claim graphs must agree (the cross-rail structural digest), and each rail's effect
    report (`--emit-effects`) must give every function that performs a volatile access the device
    footprint: `*` among both its reads and its writes.

The corpus (`units()`) is every PLACE -- a pointer-to-volatile parameter, local, file-scope pointer,
struct member and array element, and one made by a cast (`(volatile T *)raw`); a volatile member,
member array and array-of-structs field of a
plain struct; a pointer to a volatile struct, and its member array; a volatile file-scope scalar and
array; a volatile automatic scalar and array; a volatile static -- against
every ACCESS form it admits (index load/store, read-modify-write, dereference load/store, pointer
arithmetic), for each element TYPE (8/16/32/64-bit unsigned, 8/16-bit signed, float, double). One
unit per type holds all of its places, so a green run compiles eight units; a unit a rail refuses
is re-graded one form at a time, so a count is exact per form.

Out of the corpus, because neither rail lowers them (a refusal both rails agree on, not a
miscompile): increment and decrement of a volatile lvalue. A dereference of a cast integer
(`*(volatile uint32_t *)ADDR`) lowers on both rails but names no storage a harness can hand it,
so the corpus reaches the same cast through `(volatile T *)raw` instead. Two more stay
out for reasons of their own: a second subscript on a pointer element (`q[1][i]`, `pp[0][i]`), which
both rails lower as a two-dimensional index -- a miscompile that is not volatile's and is tracked
on its own -- and an initializer on a volatile aggregate, whose zero baseline the oracle also
stores element by element.

The rows (`measure`), each counted over the forms, lower is better, 0 at the bound:

    volatile.refused            forms a rail refuses (per rail)
    volatile.emit.mismatch      forms whose emitted C does not perform the original's volatile
                                accesses, in order, at their types (per rail)
    volatile.behaviour.mismatch forms whose emitted C returns a different value or leaves different
                                bytes (per rail)
    volatile.parity.mismatch    forms whose two claim graphs differ (the structural digest)
    volatile.device.missed      functions performing a volatile access that a rail's effect report
                                does not treat as a device access (per rail)

Every row needs Clang and the checkout's runtime/c (the twin is built from it): on a host without
them `measure` returns no rows, never zeros (L2).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

ROWS = (
    "volatile.refused",
    "volatile.emit.mismatch",
    "volatile.behaviour.mismatch",
    "volatile.parity.mismatch",
    "volatile.device.missed",
)

#: The element types: (tag, C type, is float). Signed types load sign-extended; floats convert.
TYPES = (
    ("u8", "uint8_t", False),
    ("u16", "uint16_t", False),
    ("u32", "uint32_t", False),
    ("u64", "uint64_t", False),
    ("i8", "int8_t", False),
    ("i16", "int16_t", False),
    ("f32", "float", True),
    ("f64", "double", True),
)

#: The access forms over an expression `X` that designates volatile storage of type `{T}`. `st` and
#: `rmw` write, the loads return their value through `R(...)`; `dst`/`dld` dereference, `pst`
#: stores through pointer arithmetic. `rmw` is integer-only (`|=` on a float is not C).
ACCESSES = {
    "st": "{X}[i & 7u] = ({T})v; return 0u;",
    "ld": "{T} x = {X}[i & 7u]; return {R};",
    "rmw": "{X}[i & 7u] |= ({T})v; return 0u;",
    "dst": "*{X} = ({T})v; return 0u;",
    "dld": "{T} x = *{X}; return {R};",
    "pst": "*({X} + (i & 7u)) = ({T})v; return 0u;",
}
#: The scalar access forms over a volatile OBJECT or MEMBER `X` of type `{T}`.
SCALAR_ACCESSES = {
    "st": "{X} = ({T})v; return 0u;",
    "ld": "{T} x = {X}; return {R};",
    "rmw": "{X} |= ({T})v; return 0u;",
}


@dataclass(frozen=True)
class Place:
    """Where the volatile qualifier sits. `params` is the entry's parameter list (before `i, v`),
    `prefix` runs first, `x` is the expression the access forms act on, `decls` go at file scope,
    `arg` is how the harness passes storage ("ptr": a `{T}` buffer, "struct": a struct buffer
    `struct_tag`, "" none), `globals` the file-scope objects the harness resets and compares."""

    params: str
    x: str
    accesses: tuple[str, ...]
    scalar: bool = False
    prefix: str = ""
    decls: str = ""
    arg: str = "ptr"
    struct_tag: str = ""
    globals: tuple[tuple[str, int], ...] = ()


#: Every place, formatted per type (`{T}` the C type, `{t}` the tag).
PLACES = {
    "param": Place("volatile {T} *p, ", "p", ("st", "ld", "rmw", "dst", "dld", "pst")),
    "local": Place(
        "{T} *b, ", "p", ("st", "ld", "rmw", "dst", "dld"), prefix="volatile {T} *p = b; "
    ),
    "vcast": Place(
        "void *raw, ",
        "p",
        ("st", "ld", "rmw", "dst", "dld", "pst"),
        prefix="volatile {T} *p = (volatile {T} *)raw; ",
    ),
    "gptr": Place(
        "{T} *b, ",
        "gp_{t}",
        ("st", "ld", "rmw", "dst", "dld", "pst"),
        prefix="gp_{t} = b; ",
        decls="volatile {T} *gp_{t};",
    ),
    "mptr": Place(
        "{T} *b, ",
        "d.regs",
        ("st", "ld", "rmw"),
        prefix="struct D_{t} d; d.regs = b; d.pad = 0u; ",
        decls="struct D_{t} {{ volatile {T} *regs; uint32_t pad; }};",
    ),
    "vmember": Place(
        "struct R_{t} *r, ",
        "r->m",
        ("st", "ld", "rmw"),
        scalar=True,
        decls="struct R_{t} {{ uint32_t pad; volatile {T} m; }};",
        arg="struct",
        struct_tag="R_{t}",
    ),
    "vstruct": Place(
        "volatile struct S_{t} *r, ",
        "r->m",
        ("st", "ld", "rmw"),
        scalar=True,
        decls="struct S_{t} {{ uint32_t pad; {T} m; }};",
        arg="struct",
        struct_tag="S_{t}",
    ),
    "vmarr": Place(
        "struct M_{t} *r, ",
        "r->arr",
        ("st", "ld", "rmw"),
        decls="struct M_{t} {{ uint32_t pad; volatile {T} arr[8]; }};",
        arg="struct",
        struct_tag="M_{t}",
    ),
    "vsarr": Place(
        "volatile struct N_{t} *r, ",
        "r->arr",
        ("st", "ld", "rmw"),
        decls="struct N_{t} {{ uint32_t pad; {T} arr[8]; }};",
        arg="struct",
        struct_tag="N_{t}",
    ),
    "vaos": Place(
        "struct E_{t} *e, ",
        "e[i & 7u].f",
        ("st", "ld", "rmw"),
        scalar=True,
        decls="struct E_{t} {{ uint32_t pad; volatile {T} f; }};",
        arg="struct",
        struct_tag="E_{t}",
    ),
    "aptr": Place(
        "{T} *b, ",
        "p",
        ("st", "ld", "rmw"),
        prefix="volatile {T} *q[2]; q[0] = b; q[1] = b; volatile {T} *p = q[1]; ",
    ),
    "vglobal": Place(
        "",
        "g_{t}",
        ("st", "ld", "rmw"),
        scalar=True,
        decls="volatile {T} g_{t};",
        arg="",
        globals=(("g_{t}", 1),),
    ),
    "vgarr": Place(
        "",
        "ga_{t}",
        ("st", "ld", "rmw", "dst", "dld"),
        decls="volatile {T} ga_{t}[8];",
        arg="",
        globals=(("ga_{t}", 8),),
    ),
    "vlocal": Place(
        "", "x0", ("st", "rmw"), scalar=True, prefix="volatile {T} x0 = ({T})i; ", arg=""
    ),
    "vlarr": Place("", "la", ("st", "rmw"), prefix="volatile {T} la[8]; ", arg=""),
    "vstatic": Place(
        "", "s0", ("st", "rmw"), scalar=True, prefix="static volatile {T} s0; ", arg=""
    ),
}


@dataclass
class Entry:
    """One form: its function, the storage the harness passes, the globals it resets."""

    name: str
    place: str
    access: str
    tag: str
    ctype: str
    arg: str
    struct_tag: str
    globals: tuple[tuple[str, int], ...]


@dataclass
class Unit:
    """A C unit and the forms it holds (one entry function per form)."""

    tag: str
    source: str
    entries: list[Entry] = field(default_factory=list)


_PRELUDE = "#include <stdint.h>\n#include <string.h>\n"


def _ret(tag: str, is_float: bool) -> str:
    """How a load returns its value as `uint32_t` without undefined behaviour: an integer converts,
    a float -- whose seeded bytes may be a NaN -- is classified by comparisons."""
    if is_float:
        return "(uint32_t)(x > 0.5) + 2u * (uint32_t)(x < -0.5) + 4u * (uint32_t)(x == x)"
    return "(uint32_t)x"


def _form(place: str, access: str, tag: str, ctype: str, is_float: bool) -> tuple[str, str, Entry]:
    """`(decls, function, entry)` for one form."""
    pl = PLACES[place]
    fmt = {"T": ctype, "t": tag}
    name = f"f_{place}_{access}_{tag}"
    table = SCALAR_ACCESSES if pl.scalar else ACCESSES
    x = pl.x.format(**fmt)
    if place == "vlocal" and access == "st":
        body = f"x0 = ({ctype})v; {ctype} x = x0; return {_ret(tag, is_float)};"
    elif place == "vlocal" and access == "rmw":
        body = f"x0 |= ({ctype})v; {ctype} x = x0; return {_ret(tag, is_float)};"
    elif place == "vstatic" and access == "st":
        body = f"s0 = ({ctype})v; {ctype} x = s0; return {_ret(tag, is_float)};"
    elif place == "vstatic" and access == "rmw":
        body = f"s0 = ({ctype})i; s0 |= ({ctype})v; {ctype} x = s0; return {_ret(tag, is_float)};"
    elif place == "vlarr" and access == "st":
        body = f"la[i & 7u] = ({ctype})v; {ctype} x = la[i & 7u]; return {_ret(tag, is_float)};"
    elif place == "vlarr" and access == "rmw":
        body = (
            f"la[i & 7u] = ({ctype})i; la[i & 7u] |= ({ctype})v; {ctype} x = la[i & 7u]; "
            f"return {_ret(tag, is_float)};"
        )
    else:
        body = table[access].format(X=x, T=ctype, R=_ret(tag, is_float))
    params = pl.params.format(**fmt) + "uint32_t i, uint32_t v"
    fn = f"uint32_t {name}({params}) {{ {pl.prefix.format(**fmt)}{body} }}"
    entry = Entry(
        name=name,
        place=place,
        access=access,
        tag=tag,
        ctype=ctype,
        arg=pl.arg,
        struct_tag=pl.struct_tag.format(**fmt),
        globals=tuple((g.format(**fmt), n) for g, n in pl.globals),
    )
    return pl.decls.format(**fmt), fn, entry


def forms() -> list[tuple[str, str, str, str, bool]]:
    """Every `(place, access, tag, ctype, is_float)` the corpus grades: `rmw` needs an integer."""
    out = []
    for tag, ctype, is_float in TYPES:
        for place, pl in PLACES.items():
            for access in pl.accesses:
                if access == "rmw" and is_float:
                    continue
                out.append((place, access, tag, ctype, is_float))
    return out


def _unit(tag: str, members) -> Unit:
    decls: list[str] = []
    fns: list[str] = []
    entries: list[Entry] = []
    for place, access, _tag, ctype, is_float in members:
        d, fn, entry = _form(place, access, tag, ctype, is_float)
        if d and d not in decls:
            decls.append(d)
        fns.append(fn)
        entries.append(entry)
    return Unit(tag, _PRELUDE + "\n".join(decls + fns) + "\n", entries)


def units() -> list[Unit]:
    """One unit per element type, holding every form of that type."""
    by_tag: dict[str, list] = {}
    for f in forms():
        by_tag.setdefault(f[2], []).append(f)
    return [_unit(tag, members) for tag, members in by_tag.items()]


def single_units(unit: Unit) -> list[Unit]:
    """The forms of `unit`, one unit each (how a refused unit is re-graded form by form)."""
    return [
        _unit(
            unit.tag,
            [(e.place, e.access, e.tag, e.ctype, dict((t, f) for t, _c, f in TYPES)[e.tag])],
        )
        for e in unit.entries
    ]


# --- the two rails -------------------------------------------------------------------------------


@dataclass
class RailResult:
    """What one rail made of a unit: its emitted C (every function), its structural digest, its
    effect report, or the reason it refused."""

    refused: str = ""
    emitted: str = ""
    digest: str = ""
    effects: str = ""


def oracle(unit: Unit) -> RailResult:
    from bcir.frontends.cfront import compile_unit  # noqa: PLC0415
    from bcir.frontends.cfront.escape import effects_report  # noqa: PLC0415
    from bcir.verify import cfront_structural_digest  # noqa: PLC0415

    try:
        res = compile_unit(unit.source, check_clang=False)
    except Exception as exc:  # noqa: BLE001 -- a refusal is a verdict, not a traceback (L1)
        return RailResult(refused=f"{type(exc).__name__}: {exc}")
    if not res.is_clean:
        diags = getattr(res, "diagnostics", None) or []
        return RailResult(
            refused="; ".join(str(getattr(d, "message", d)) for d in diags[:3]) or "dirty"
        )
    return RailResult(
        emitted="\n".join(res.emitted[name] for name in res.lowered.functions),
        digest=f"{cfront_structural_digest(res.lowered):016x}",
        effects=effects_report(res.lowered, res.escape),
    )


def twin(unit: Unit, bcir_cc: str) -> RailResult:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "unit.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(unit.source)
        out = {}
        for flag in ("--emit-c", "--emit-claimgraph", "--emit-effects"):
            try:
                run = subprocess.run(
                    [bcir_cc, flag, path], capture_output=True, text=True, timeout=120
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return RailResult(refused=f"{flag}: {exc}")
            if run.returncode != 0:
                lines = run.stderr.strip().splitlines() or [f"rc={run.returncode}"]
                return RailResult(refused=lines[-1].replace(path, "unit.c"))
            out[flag] = run.stdout
    m = re.search(r"digest=([0-9a-f]{16})", out["--emit-claimgraph"])
    # the driver makes `--emit-c` a self-contained unit by including the quarantine runtime header
    # when an access is masked; the judges inline that ABI (`_bounds_guard`), as for the oracle's emit
    return RailResult(
        emitted=out["--emit-c"].replace('#include "bcir_quarantine.h"\n', ""),
        digest=m.group(1) if m else "",
        effects=out["--emit-effects"],
    )


# --- the judges ----------------------------------------------------------------------------------


def find_clang() -> str | None:
    """Clang: the IR judge reads LLVM's own volatile flags, so only Clang will do."""
    return shutil.which("clang")


def _bounds_guard() -> str:
    from bcir.tests.test_c_cfront import _BOUNDS_GUARD  # noqa: PLC0415

    return _BOUNDS_GUARD


_VOLATILE_OP = re.compile(r"\b(load|store) volatile (\S+?),? ")


def volatile_ops(clang: str, source: str) -> dict[str, list[tuple[str, str]]]:
    """Every defined function of `source` -> its volatile loads and stores, in order, with their
    LLVM types (Clang -O2, inlining off, so each function keeps its own accesses)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "unit.c")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(source)
        run = subprocess.run(
            [clang, "-std=gnu11", "-O2", "-fno-inline", "-w", "-S", "-emit-llvm", "-o", "-", path],
            capture_output=True,
            text=True,
            timeout=300,
        )
    if run.returncode != 0:
        raise RuntimeError("clang: " + (run.stderr.strip().splitlines() or ["?"])[-1])
    ops: dict[str, list[tuple[str, str]]] = {}
    current = None
    for line in run.stdout.splitlines():
        if line.startswith("define "):
            m = re.search(r"@([A-Za-z_][A-Za-z0-9_]*)\(", line)
            current = m.group(1) if m else None
            if current:
                ops[current] = []
        elif line.startswith("}"):
            current = None
        elif current:
            for kind, ty in _VOLATILE_OP.findall(line):
                ops[current].append((kind, ty))
    return ops


def _harness(unit: Unit, emitted: str) -> str:
    """A program running each entry original-then-emitted from the same seeded memory."""
    lines = [
        _PRELUDE + "#include <stdio.h>\n" + _bounds_guard(),
        unit.source,
        emitted,
        "static uint64_t S = 0x9E3779B97F4A7C15u;",
        "static uint32_t rng(void) { S = S * 6364136223846793005u + 1442695040888963407u;"
        " return (uint32_t)(S >> 32); }",
        "static void fill(void *p, size_t n) { unsigned char *q = p;"
        " for (size_t k = 0; k < n; k++) q[k] = (unsigned char)rng(); }",
        "int main(void) {",
        "  int bad = 0;",
    ]
    for e in unit.entries:
        if e.arg == "ptr":
            store = f"{e.ctype} a[64], b[64];"
            argsa, argsb = "a, ", "b, "
        elif e.arg == "struct":
            store = f"struct {e.struct_tag} a[8], b[8];"
            argsa, argsb = "a, ", "b, "
        else:
            store = "unsigned char a[1], b[1];"
            argsa = argsb = ""
        lines.append("  {")
        lines.append(f"    {store}")
        for g, n in e.globals:
            lines.append(
                f"    unsigned char g0_{g}[sizeof {g}], g1_{g}[sizeof {g}], g2_{g}[sizeof {g}];"
            )
        lines.append("    for (int t = 0; t < 32 && !bad; t++) {")
        lines.append("      fill(a, sizeof a); memcpy(b, a, sizeof a);")
        lines.append("      uint32_t i = rng() % 200u, v = rng();")
        for g, n in e.globals:
            lines.append(
                f"      fill(g0_{g}, sizeof g0_{g});"
                f" for (size_t k = 0; k < sizeof g0_{g}; k++) ((volatile unsigned char *)&{g})[k] = g0_{g}[k];"
            )
        lines.append(f"      uint32_t r1 = {e.name}({argsa}i, v);")
        for g, n in e.globals:
            lines.append(
                f"      for (size_t k = 0; k < sizeof g1_{g}; k++) g1_{g}[k] = ((volatile unsigned char *)&{g})[k];"
                f" for (size_t k = 0; k < sizeof g0_{g}; k++) ((volatile unsigned char *)&{g})[k] = g0_{g}[k];"
            )
        lines.append(f"      uint32_t r2 = bcir_{e.name}({argsb}i, v);")
        cmp = ["r1 != r2", "memcmp(a, b, sizeof a)"]
        for g, n in e.globals:
            lines.append(
                f"      for (size_t k = 0; k < sizeof g2_{g}; k++) g2_{g}[k] = ((volatile unsigned char *)&{g})[k];"
            )
            cmp.append(f"memcmp(g1_{g}, g2_{g}, sizeof g1_{g})")
        lines.append(
            f'      if ({" || ".join(cmp)}) {{ printf("MISMATCH {e.name}\\n"); bad = 1; }}'
        )
        lines.append("    }")
        lines.append("    bad = 0;")
        lines.append("  }")
    lines.append('  printf("DONE\\n"); return 0; }')
    return "\n".join(lines) + "\n"


def behaviour_mismatches(clang: str, unit: Unit, emitted: str) -> set[str]:
    """The entries of `unit` whose emitted C behaves differently from the original (every entry,
    when the harness does not build or does not finish)."""
    everything = {e.name for e in unit.entries}
    with tempfile.TemporaryDirectory() as d:
        src, exe = os.path.join(d, "h.c"), os.path.join(d, "h")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write(_harness(unit, emitted))
        b = subprocess.run(
            [clang, "-std=gnu11", "-O2", "-w", src, "-o", exe],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if b.returncode != 0:
            return everything
        try:
            r = subprocess.run([exe], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            return everything
    if "DONE" not in r.stdout:
        return everything
    return {line.split()[1] for line in r.stdout.splitlines() if line.startswith("MISMATCH ")}


def emit_mismatches(clang: str, unit: Unit, emitted: str) -> set[str]:
    """The entries whose emitted twin does not perform the original's volatile accesses, in order,
    at their types."""
    # the emitted functions are `static` and nothing calls them here: take their addresses in a
    # used table, or Clang drops them before the comparison
    keep = ", ".join(f"(void *)&bcir_{e.name}" for e in unit.entries)
    table = f"static void *const bcir_keep_[] __attribute__((used)) = {{ {keep} }};\n"
    try:
        ops = volatile_ops(
            clang, _PRELUDE + _bounds_guard() + "\n" + unit.source + "\n" + emitted + "\n" + table
        )
    except (RuntimeError, OSError, subprocess.TimeoutExpired):
        return {e.name for e in unit.entries}
    return {
        e.name
        for e in unit.entries
        if ops.get(e.name) != ops.get(f"bcir_{e.name}") or not ops.get(e.name)
    }


_EFFECT_LINE = re.compile(r"^fn=(\S+) reads=(\S*) writes=(\S*)$")


def device_missed(unit: Unit, effects: str) -> set[str]:
    """The entries whose effect footprint is not a device access's: a device access reads and writes
    unknown memory, so `*` must be in both sets (beside any object the function also names)."""
    seen = {}
    for line in effects.splitlines():
        m = _EFFECT_LINE.match(line.strip())
        if m:
            seen[m.group(1)] = (set(m.group(2).split(",")), set(m.group(3).split(",")))
    return {
        e.name
        for e in unit.entries
        if e.name not in seen or "*" not in seen[e.name][0] or "*" not in seen[e.name][1]
    }


# --- the rows ------------------------------------------------------------------------------------


def _grade(unit: Unit, bcir_cc: str, clang: str) -> dict[str, set]:
    """Per-row sets of failing `(rail, entry)` for one unit (both rails accepted it)."""
    py, tw = oracle(unit), twin(unit, bcir_cc)
    out = {row: set() for row in ROWS}
    for rail, res in (("oracle", py), ("twin", tw)):
        if res.refused:
            out["volatile.refused"] |= {(rail, e.name) for e in unit.entries}
            continue
        out["volatile.emit.mismatch"] |= {
            (rail, n) for n in emit_mismatches(clang, unit, res.emitted)
        }
        out["volatile.behaviour.mismatch"] |= {
            (rail, n) for n in behaviour_mismatches(clang, unit, res.emitted)
        }
        out["volatile.device.missed"] |= {(rail, n) for n in device_missed(unit, res.effects)}
    if not py.refused and not tw.refused and py.digest != tw.digest:
        out["volatile.parity.mismatch"] |= {("both", e.name) for e in unit.entries}
    return out


def available(bcir_cc: str | None = None) -> bool:
    """Whether this host can grade: Clang, and a twin (given, or buildable from runtime/c)."""
    return bool(find_clang()) and (bool(bcir_cc) or os.path.isdir(_runtime_c()))


def _runtime_c() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runtime", "c"
    )


def failures(bcir_cc: str | None = None, which=None) -> dict[str, set]:
    """Per-row sets of failing `(rail, entry)` over the corpus (or the units `which`). A unit either
    rail refuses is re-graded one form at a time, so a refusal costs only its own forms."""
    from bcir.tests.escape_fixtures import build_bcir_cc  # noqa: PLC0415

    clang = find_clang()
    if not clang:
        raise RuntimeError("no clang: the volatile rows are graded by Clang")
    if bcir_cc is None:  # grade while the directory the twin was built in still exists
        with tempfile.TemporaryDirectory() as d:
            return failures(build_bcir_cc(d), which)
    total = {row: set() for row in ROWS}
    for unit in which if which is not None else units():
        graded = _grade(unit, bcir_cc, clang)
        if graded["volatile.refused"]:
            for single in single_units(unit):
                for row, bad in _grade(single, bcir_cc, clang).items():
                    total[row] |= bad
        else:
            for row, bad in graded.items():
                total[row] |= bad
    return total


def measure(bcir_cc: str | None = None) -> dict[str, float]:
    """The five rows, or `{}` when the host cannot grade (never zeros by default, L2)."""
    if not available(bcir_cc):
        return {}
    return {row: float(len(bad)) for row, bad in failures(bcir_cc).items()}
