# BCIR C Frontend — User Guide (`bcir-cfront`)

`bcir-cfront` is the C-frontend driver: it compiles a useful subset of C23 through the BCIR claim
graph, verifies the result against the R1–R18 laws, emits behaviour-equivalent C, and reports
Clang-style diagnostics. What it cannot yet compile, it cleanly hands off (the LLVM-backend fallback
contract) — so it behaves like a *verified-subset* compiler, not a research toy.

This guide is the practical quickstart + capability/limits reference. For the numbers (test counts,
coverage) see [`STATUS.md`](STATUS.md); for where it sits in the project, the
[master roadmap](BCIR_MASTER_ROADMAP.md); for the Clang comparison, [`CLANG_COMPARISON.md`](CLANG_COMPARISON.md).

## Quickstart

```sh
# compile a file (verified C + R1–R18 status to stdout)
python -m bcir.frontends.cfront hello.c

# syntax/semantic check only — Clang-style diagnostics, no output
python -m bcir.frontends.cfront -fsyntax-only hello.c

# machine-readable diagnostics (for editors / CI)
python -m bcir.frontends.cfront --emit-json hello.c

# lay the types out for another target ABI
python -m bcir.frontends.cfront --target x86_64-windows hello.c

# graceful degradation: report a fallback-to-LLVM signal instead of erroring out
python -m bcir.frontends.cfront --fallback hello.c
```

The driver lives in [`bcir/frontends/cfront/__main__.py`](../bcir/frontends/cfront/__main__.py); the
library entry points are `compile_unit`, `diagnose`, and `compile_with_fallback`.

## Command-line options

| Option | Meaning |
| --- | --- |
| `-I <dir>` | add a `#include` search-path directory (repeatable; the source's own dir is always searched) |
| `-D name[=val]` | predefine an object macro (`val` defaults to 1) |
| `-U name` | undefine a predefined / `-D` macro |
| `-std=<std>` | language standard: `c23`/`c2x` (default), `c17`, `c11` |
| `-E` | preprocess only — print the expanded translation unit |
| `--target <abi>` | the target data model the unit is laid out for (default `x86_64-linux`) |
| `-fsyntax-only` | parse + check only; print diagnostics, emit no compiled output |
| `--emit-json` | print diagnostics as a machine-readable JSON array |
| `--fallback` | report a fallback-to-LLVM signal (exit 2) for unsupported constructs instead of erroring |
| `-o <file>` | write output to `<file>` instead of stdout |
| `--explain` | also print the per-function plan/explain record |
| `--selfcheck` | print the generated dual-rail self-check harness |

Exit codes: `0` clean, `1` a diagnostic (error), `2` a usage error or a fallback-to-LLVM signal.

## Diagnostics

Errors are reported in the Clang layout — a `file:line:col: severity: message` banner, the source
line, and a `^~~~` caret — with parser **error recovery** (one run reports several errors),
**fix-it** hints for missing punctuation, **`In file included from …`** frames for errors in headers,
and a `--emit-json` machine-readable form. The engine is
[`bcir/frontends/cfront/diagnostics.py`](../bcir/frontends/cfront/diagnostics.py).

```
hello.c:2:10: error: use of undeclared identifier 'undeclared'
    return undeclared + 1;
           ^
```

## The target ABI matrix

`--target` selects the data model the frontend lays types out for
([`abi.py`](../bcir/frontends/cfront/abi.py)). The named targets:

| Target | Data model | `long` | pointer | `long double` |
| --- | --- | --- | --- | --- |
| `x86_64-linux`, `aarch64-linux`, `riscv64-linux` | LP64 | 8 | 8 | 16 |
| `x86_64-windows` | LLP64 | 4 | 8 | 8 |
| `i386-linux` | ILP32 | 4 | 4 | 12 |

So `struct { long a; char b; }` is 16 bytes on LP64 and 8 bytes on LLP64 / ILP32. The host target's
output is validated against the host C compiler (behaviour-equivalence); a cross-target layout is not
byte-compatible with the host compiler, so its equivalence check is reported as `skip:cross-target`
(the layout is conformance-checked instead). Float math and calling conventions are delegated to the
backend.

## The LLVM-backend fallback contract

BCIR compiles the supported, fully-verified subset; for anything outside it, `--fallback` (library:
`compile_with_fallback`) returns a result whose `needs_fallback` is set and whose `fallback` names the
rejecting stage + reason — the signal for a driver to route the unit to the LLVM backend rather than
fail. Without `--fallback`, an unsupported construct is a normal diagnostic.

```
fb.c: fallback to LLVM backend: lower: static initializer is not a constant expression
```

## What's supported

- Fixed-width and core integer types, `_Bool`/`char`, `void`, `float`/`double`/`long double`, pointers,
  arrays, `struct`/`union` (Clang-compatible layout, per target), `enum`, `typedef`.
- Integer + IEEE-754 floating arithmetic and comparisons, casts and the usual arithmetic conversions,
  `sizeof`/`_Alignof`, bitfields, `<math.h>` library calls.
- Functions, the call graph (R18: callee resolution, no recursion), and inter-procedural summary reuse.
- String/character literals (with prefixes), `static` locals, file-scope globals, `volatile` (MMIO).
- The preprocessor: `#include`/`#embed`, conditionals, object/function-like + variadic macros, the
  predefined macros, `#line`, `_Pragma`, and the `__has_*` feature-test operators.

## Known limits

These are reported as diagnostics, or — with `--fallback` — as a fallback-to-LLVM signal:

- Non-constant `static`/global initializers; constructs beyond the L1–L6 statement subset.
- 64-bit-integer **results** of a few `<math.h>` functions and pointer out-params are supported, but a
  general 64-bit *value* model and Windows/ILP32 *code generation* (vs. layout) are not.
- The i386 in-struct `double`-alignment quirk is not modelled (the `long`/pointer/`long double` data
  axes are).
- Cross-target builds are layout-only here; running them needs a cross toolchain.

When in doubt, run `-fsyntax-only` (or `--fallback`) — the frontend names exactly what it can't do.
