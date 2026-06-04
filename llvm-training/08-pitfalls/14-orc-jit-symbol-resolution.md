# Pitfall 14 — ORC JIT Symbol Resolution

## The error

Common ORC/LLJIT lookup failures look like:

```text
Symbols not found: [ foo ]
```

Or, when using a mangled platform symbol accidentally:

```text
Symbols not found: [ _foo ]
```

A worse version is a successful lookup followed by a crash because the address
was cast to the wrong function pointer type.

## Minimal reproducer

IR module:

```llvm
define i32 @entry() {
  ret i32 7
}
```

C++ outline with the bug:

```cpp
auto Sym = JIT->lookup("_entry");  // ❌ wrong spelling on many targets
```

Or the opposite bug in lower-level ORC code: a platform expects a mangled symbol,
but the lookup uses the IR spelling without going through the data-layout-aware
mangler.

Another common failure is forgetting process symbols:

```llvm
declare i32 @puts(ptr)

define i32 @entry(ptr %s) {
  %r = call i32 @puts(ptr %s)
  ret i32 %r
}
```

If the JITDylib has no generator for host-process symbols, lookup/materialization
can fail when `puts` is needed.

## Why it happens

ORC resolves symbols in JITDylibs, not in one flat magical namespace. Symbol
names also pass through target object-file conventions. C identifiers, LLVM IR
function names, object symbols, and executor symbols are related, but they are
not always identical strings.

`LLJIT` hides much of this for simple examples, but lower-level ORC code still
requires deliberate setup: data layout, symbol mangling, JITDylib search order,
absolute symbols, dynamic library generators, and resource ownership.

## Fix pattern

For `LLJIT`, start with the simple spelling for exported IR names:

```cpp
auto Sym = JIT->lookup("entry");
if (!Sym)
  return Sym.takeError();
```

For lower-level APIs, mangle with the JIT data layout instead of hand-writing
prefixes:

```cpp
llvm::orc::MangleAndInterner Mangle(ES, DL);
auto Sym = ES.lookup({&JD}, Mangle("entry"));
```

When JITed code calls host-process functions, install the appropriate dynamic
library search generator for the process and data layout. Keep JITDylib search
order explicit when multiple modules can define the same name.

Finally, cast the result to the exact ABI type:

```cpp
using EntryFn = int (*)();
auto *Entry = Sym->getAddress().toPtr<EntryFn>();
```

## BCIR-relevant note

A BCIR JIT may need symbols for lifted helper routines, runtime validators,
external library functions, and generated entry points. Put helpers in a known
JITDylib, decide whether lifted binary symbols keep original names or receive a
namespace prefix, and mangle through ORC rather than by string concatenation.
Otherwise, two recovered objects can accidentally shadow each other or fail to
resolve runtime support symbols.

## See also

- [`../12-backend-jit/03-orc-jit.md`](../12-backend-jit/03-orc-jit.md) — ORC/LLJIT architecture and lookup
- [`../12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md) — code generation context
- [`../01-syntax/01-modules-functions-blocks.md`](../01-syntax/01-modules-functions-blocks.md) — modules and symbol names
