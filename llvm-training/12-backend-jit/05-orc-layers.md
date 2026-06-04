# ORC Layers, Symbols, and JITLink

[`03-orc-jit.md`](03-orc-jit.md) introduced `LLJIT` as the easiest way to use
ORC. This follow-up opens the wrapper and names the lower-level pieces you will
see when debugging a non-trivial JIT: the `ExecutionSession`, `JITDylib`s,
compile/object/transform layers, materialization responsibilities, symbol
interning, and the handoff from generated object files to JITLink relocations.

Read this chapter together with [`04-mc-and-relocations.md`](04-mc-and-relocations.md):
ORC schedules compilation and symbol lookup, while MC/JITLink provide the
concrete object symbols, sections, edges, fixups, and relocation records that
must be resolved before bytes can be called.

## Layer stack at a glance

A typical ORC JIT stack built by `LLJIT` or by a custom `KaleidoscopeJIT` looks
like this:

```text
client code
  -> ExecutionSession lookup("name")
  -> JITDylib symbol table and search order
  -> materialization unit / MaterializationResponsibility
  -> optional IR transform layer
  -> IR compile layer
  -> object transform layer, if configured
  -> object/link layer
  -> JITLink / RuntimeDyld
  -> executable memory + resolved relocations
```

The arrows are conceptual, not always direct calls. A lookup can trigger lazy
materialization; materialization can be concurrent; and a layer may hand work to
another layer without exposing the intermediate object to client code.

## `ExecutionSession`

An `ExecutionSession` is the session-wide coordinator for ORC. It owns or
coordinates:

- interned symbol strings used by definitions and lookups;
- asynchronous error reporting and dispatch state;
- `JITDylib` creation and lookup across dylibs;
- session lifetime for materialization work;
- access to the executor process in newer ORC configurations.

The most important practical rule is: do not treat symbol names as ordinary
`std::string`s once you cross into lower-level ORC APIs. Intern names through the
session, usually through helper APIs such as `MangleAndInterner`, so every layer
compares the same canonical symbol-string identity.

## `JITDylib`

A `JITDylib` is ORC's symbol namespace. It is analogous to a dynamic library in
that it owns definitions and participates in a search order, but it is not
limited to symbols from one file. A `JITDylib` can contain:

- IR modules waiting to be compiled;
- already-emitted object files;
- absolute symbols supplied by the embedding process;
- generators that search the host process or libraries on demand;
- aliases, reexports, weak definitions, and lazily materialized symbols.

Use multiple `JITDylib`s when you need library-like isolation, unloadable groups,
or explicit symbol precedence. Use one main `JITDylib` for simple tools. When a
lookup fails, record which `JITDylib` was searched and which generators were
attached before inspecting the object file.

## Compile layers

Compile layers accept a program representation and eventually emit object code.
The common IR compile layer is `IRCompileLayer`:

```text
ThreadSafeModule
  -> IR transforms, if any
  -> target machine / compiler
  -> object buffer
  -> object layer
```

Important compile-layer responsibilities:

- verify that the module's data layout and target triple match the JIT target;
- run any configured optimization or instrumentation pipeline;
- lower IR through the backend pipeline described in
  [`01-codegen-pipeline.md`](01-codegen-pipeline.md);
- emit an object file containing sections, symbol tables, and relocations.

If compilation fails before an object exists, inspect IR, target features, data
layout, and pass diagnostics. If compilation succeeds but lookup or call fails,
move down to the object/link layer and the MC artifacts from
[`04-mc-and-relocations.md`](04-mc-and-relocations.md).

## Object and link layers

Object layers accept object buffers. Link layers allocate memory, resolve symbols,
apply relocations, register exception frames where needed, and publish the final
symbol addresses. Modern ORC configurations commonly use `ObjectLinkingLayer`,
which is built on JITLink.

The object/link layer is where symbolic machine-code dependencies become
runtime addresses:

```text
object file symbols + sections + relocations
  -> JITLink LinkGraph blocks, symbols, and edges
  -> allocation in executor memory
  -> relocation/fixup application
  -> symbol publication
```

When a missing symbol names a runtime helper, libcall, or platform-prefixed
symbol, the object/link layer is usually where the unresolved relocation is
reported. Use `llvm-nm`, `llvm-objdump -dr`, and `llvm-readobj --relocations
--symbols` as described in [`04-mc-and-relocations.md`](04-mc-and-relocations.md)
to see exactly which object artifact forced the lookup.

## Transform layers

Transform layers are pass-through layers that inspect or rewrite a representation
before forwarding it to the next layer.

Common transform points:

| Layer kind | Input | Typical use |
| --- | --- | --- |
| IR transform layer | `ThreadSafeModule` | Run optimization passes, add instrumentation, validate declarations, rewrite helper names. |
| object transform layer | object buffer | Dump objects for debugging, sign code, rewrite sections, collect relocation statistics. |
| link-graph pass/plugin | JITLink graph | Apply platform fixups, GOT/PLT/stub generation, memory protection policy, edge validation. |

For training and debugging, an object transform that writes every object buffer
to `/tmp` is often the fastest way to connect an ORC error to the MC-level symbol
and relocation tables.

## Materialization responsibility

ORC separates declaring that a symbol will exist from doing the work needed to
produce it. A materialization unit promises definitions to a `JITDylib`. When a
lookup needs those definitions, ORC calls materialization code with a
`MaterializationResponsibility`.

Think of `MaterializationResponsibility` as the work order for one triggered
materialization:

- it identifies the symbols this materialization is responsible for producing;
- it lets the layer notify ORC when symbols are resolved to addresses;
- it lets the layer notify ORC when symbols are fully emitted and ready;
- it carries failure reporting so dependent lookups receive a useful error;
- it can delegate or replace responsibility when materialization is split.

A layer that forgets to resolve, emit, or fail its responsibility can leave
lookups hanging or produce confusing cascading errors. A layer that resolves the
wrong symbol name can make the link graph look correct while the `JITDylib`
symbol table still lacks the symbol that clients requested.

## Symbol resolution and interning

ORC symbol lookup has two related naming concerns:

1. **Mangled object names.** The object file may use target-specific naming, such
   as leading underscores on some platforms or C++ ABI mangling.
2. **Interned ORC names.** ORC stores symbol names as interned strings owned by
   the `ExecutionSession`, so definitions and lookups should use the same
   canonical symbol identity.

For lower-level APIs, create a mangle-and-intern helper from the session and data
layout, then use it consistently for declarations, absolute symbols, and lookups.
For C++ runtime helpers, prefer `extern "C"` wrappers if IR should call simple
unmangled names.

Symbol-resolution checklist:

- Confirm the IR declaration name and linkage.
- Confirm the emitted object symbol name with `llvm-nm`.
- Confirm the relocation target with `llvm-objdump -dr` or `llvm-readobj`.
- Confirm the `JITDylib` search order and attached symbol generators.
- Confirm host-process symbols were registered before materialization needs them.

## Interaction with MC, JITLink, and relocations

The compile layer reaches the backend pipeline, where LLVM lowers IR to
`MachineInstr`, then `MCInst`, then object bytes. The MC layer records unresolved
values as symbols, fixups, and relocations. ORC does not invent those relocation
dependencies; it receives them from the object file generated by MC.

JITLink then reads the object into a link graph:

| MC/object concept | JITLink concept | Why it matters in ORC |
| --- | --- | --- |
| section or subsection | block / graph section | Determines allocation, permissions, and layout. |
| symbol table entry | graph symbol | Publishes definitions and names external dependencies. |
| relocation/fixup | edge | Describes how to patch bytes once a target address is known. |
| addend/expression | edge addend/kind | Captures PC-relative, absolute, GOT, branch, or target-specific semantics. |

If an ORC error says a symbol cannot be found, the relocation table is often the
backend's concrete dependency list. If an ORC error says a relocation or edge
kind is unsupported, the symbol may exist but the JITLink target backend does not
know how to apply that relocation kind for the current object format or CPU.

## Troubleshooting ORC-layer failures

| Symptom | Likely layer | Inspect these artifacts | Common fix |
| --- | --- | --- | --- |
| `Failed to materialize symbols` after adding IR | compile or transform layer | IR verifier output, target triple, data layout, pass diagnostics | Set the module data layout from the JIT, fix invalid IR, or disable the transform that introduced invalid IR. |
| `Symbols not found: [ foo ]` | `JITDylib` lookup or object/link layer | IR declarations, `llvm-nm` symbol table, `llvm-objdump -dr` relocation target | Use mangle-and-intern consistently, export an `extern "C"` helper, or add a host-process generator to the searched `JITDylib`. |
| Lookup uses `foo` but object defines `_foo` or a C++ mangled name | symbol interning/mangling boundary | data layout global prefix, `llvm-nm --demangle`, source helper declaration | Lookup the mangled-and-interned name or change the helper to an unmangled exported wrapper. |
| Undefined libcall such as `memcpy`, `__divti3`, or atomic helper | backend lowering plus object/link layer | generated object relocations, target features, runtime library exports | Provide the runtime symbol, link an object/library defining it, or change lowering/options to avoid that libcall. |
| Relocation/edge kind unsupported | JITLink target backend | `llvm-readobj --relocations`, object format, target triple/CPU/features | Use a supported object format/target, update JITLink support, or change code model/relocation model. |
| Symbol exists but lookup still fails | `JITDylib` search order | `JITDylib` definitions, generators, reexports, weak/strong binding | Add the provider to the searched `JITDylib`, adjust search order, or reexport the symbol. |
| Crash after converting address to function pointer | client ABI boundary | IR function type, C++ function pointer type, calling convention, argument ABI | Match the exact signature and calling convention; avoid casting data symbols as code. |
| Code unload/removal breaks later calls | resource tracking/lifetime | `ResourceTracker` ownership, outstanding function pointers, dependent dylibs | Keep resources alive while pointers may be called, or design explicit invalidation for hot reload. |
| Lazy lookup never completes or hangs | materialization responsibility | layer logs around `resolve`, `emit`, and `failMaterialization` | Ensure each responsibility is resolved, emitted, delegated, or failed on every path. |
| Platform-specific failure registering EH frames or stubs | object/link layer platform support | sections such as `.eh_frame`, JITLink plugin logs, object headers | Configure the platform/link-layer plugins required by the executor and object format. |

## Debugging workflow

1. Decide whether the failure happened before object emission, during lookup, or
   while linking/applying relocations.
2. If no object was emitted, debug IR, target setup, and transform passes.
3. If an object was emitted, save it and inspect symbols and relocations with the
   commands in [`04-mc-and-relocations.md`](04-mc-and-relocations.md).
4. Match each unresolved relocation target to a `JITDylib` definition or symbol
   generator.
5. Check that published ORC names were mangled and interned with the same
   `ExecutionSession` and data layout used for lookup.
6. For unsupported relocation failures, switch from symbol debugging to JITLink
   target/backend debugging: object format, relocation kind, CPU features, code
   model, and platform plugins.

## How this extends the `LLJIT` chapter

Use [`03-orc-jit.md`](03-orc-jit.md) for the high-level API sequence: create an
`LLJIT`, add a `ThreadSafeModule`, look up a symbol, and call it. Use this
chapter when you need to explain which lower-level component owns a failure or
when a custom JIT adds optimization, lazy compilation, object dumping, remote
execution, custom memory management, or multiple symbol namespaces.
