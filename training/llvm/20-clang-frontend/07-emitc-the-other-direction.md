# `emitc`: the other direction

Every other chapter in this directory goes one way — C in, IR out. The `emitc` dialect goes
back, and reading its output is a good way to find out what you actually believed about the
IR you have been reading.

## The example

[`examples/emitc-mlir-to-c.mlir`](examples/emitc-mlir-to-c.mlir) is as small as it gets:

```mlir
func.func @add(%a: i32, %b: i32) -> i32 {
  %c = arith.addi %a, %b : i32
  return %c : i32
}
```

Convert it and translate it:

```console
$ mlir-opt emitc-mlir-to-c.mlir \
    --pass-pipeline='builtin.module(convert-arith-to-emitc,convert-func-to-emitc)' \
  | mlir-translate --mlir-to-cpp
```
```cpp
int32_t add(int32_t v1, int32_t v2) {
  uint32_t v3 = (uint32_t) v1;
  uint32_t v4 = (uint32_t) v2;
  uint32_t v5 = v3 + v4;
  int32_t v6 = (int32_t) v5;
  return v6;
}
```

One addition became four statements and a round trip through `uint32_t`. That is not
clumsiness; it is the only correct translation.

## Why it goes through unsigned

`arith.addi` is **sign-agnostic and wraps**. Two's-complement addition does not care which
signedness you had in mind, and overflow is defined: it wraps.

C's `+` on `int32_t` is **undefined on overflow**. A C compiler is entitled to assume it
never happens, and to optimise on that assumption — which is the whole subject of `nsw` in
[`../08-pitfalls/`](../08-pitfalls) and the reason `arith.addi` carries no `nsw` by default.

So a direct `v1 + v2` would be *wrong*: it would take an operation whose overflow is
defined and emit one whose overflow is undefined, handing the C compiler a licence the MLIR
never granted. Unsigned arithmetic in C **is** defined to wrap, so the conversion casts to
`uint32_t`, adds there, and casts back. The casts are free at runtime and the semantics are
preserved exactly.

**This is the lesson worth taking away from `emitc` generally.** Emitting a higher-level
language is not printing; it is a lowering into a target whose undefined behaviour is
different from yours, and every place the two disagree costs a cast.

## Two spellings, again

The converted MLIR prints like this:

```mlir
emitc.func @add(%arg0: i32, %arg1: i32) -> i32 {
  %0 = cast %arg0 : i32 to ui32
  %2 = add %0, %1 : (ui32, ui32) -> ui32
  return %3 : i32
}
```

Note the inner operations: bare `cast`, `add`, `return`. Their real names are
`emitc.cast`, `emitc.add` and `emitc.return` — you only see those under
`--mlir-print-op-generic`, because `emitc.func`'s custom printer drops the prefix inside its
own region. This is exactly the asymmetry
[`../24-mlir-infrastructure/01-regions-blocks-and-the-two-syntaxes.md`](../24-mlir-infrastructure/01-regions-blocks-and-the-two-syntaxes.md)
describes, met in the wild: a tool that greps for `emitc.add` in the default output finds
nothing, and concludes wrongly that the conversion did not happen.

## What the emission is not

The C++ above does not compile on its own:

```console
$ clang++ -std=c++17 -c out.cpp
error: unknown type name 'int32_t'
```

`mlir-translate --mlir-to-cpp` emits a **fragment**, not a translation unit. There is no
`#include <cstdint>`, no header guard, no namespace — prepend the include and it compiles
clean. Whatever embeds the output is responsible for the surrounding file, which is the
right division of labour but surprises people who expect a compiler-shaped tool to produce
a compiler-shaped artifact.

## Pitfalls checklist

- Do not read the unsigned round trip as inefficiency. Removing it would change the
  program's meaning on overflow.
- Do not grep the default output for `emitc.add`. Inside `emitc.func` the prefix is not
  printed; use `--mlir-print-op-generic` if you need the real names.
- Do not expect `--mlir-to-cpp` output to be a compilable file. It is a fragment and needs
  at least `<cstdint>`.
- Do not assume every MLIR construct has an `emitc` form. The dialect covers what maps onto
  C cleanly; a construct with no C analogue has no conversion, and the pass will leave it
  alone rather than invent one.

## Checks

[`examples/emitc-mlir-to-c.mlir`](examples/emitc-mlir-to-c.mlir) is registered in
[`../autograder/mlir-examples.json`](../autograder/mlir-examples.json) as a Tier 3
conversion: the registry runs the two conversion passes and requires `emitc.func @add` and
`ui32` to appear while forbidding `arith.addi` and `func.func` — so the unsigned round trip
described above is a checked property, not a remembered one.

**Stated rather than implied:** the C++ emission itself is *not* re-run by a gate. The
registry's checks cover MLIR and LLVM IR, and adding a C++ compile step would need a host
guarantee the base training job does not make. The C++ shown above is reproduced from the
command printed with it, and the conversion that produces its shape is gated; the final
`mlir-translate --mlir-to-cpp` step is not. That is a real hole in the coverage of this
chapter, and naming it is better than leaving a reader to assume otherwise.
