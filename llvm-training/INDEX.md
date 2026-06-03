# INDEX — Topic / Symbol → File

Agent: this is your entry point. Find your topic, jump to the file.

## By topic

| Topic | File |
|---|---|
| What LLVM IR is, big picture | `00-foundations/01-what-is-llvm-ir.md` |
| SSA form, phi nodes | `00-foundations/02-ssa.md` |
| LLVM IR vs assembly, vs GIMPLE/CIL/SPIR-V | `00-foundations/03-ir-vs-asm-vs-other-irs.md` |
| Modules, functions, basic blocks | `01-syntax/01-modules-functions-blocks.md` |
| Instruction format, operands | `01-syntax/02-instruction-format.md` |
| Comments (`;`), metadata (`!N`, `!{...}`) | `01-syntax/03-comments-metadata.md` |
| Integer types `iN` | `02-types/01-primitive-types.md` |
| `float`, `double`, `half`, `bfloat`, `fp128` | `02-types/01-primitive-types.md` |
| `void`, `ptr`, `label`, `token`, `metadata` | `02-types/01-primitive-types.md` |
| Struct, array, vector | `02-types/02-composite-types.md` |
| Opaque types, opaque pointers | `02-types/03-opaque-and-pointer-types.md` |
| Integer constants (`i32 42`) | `03-constants/01-integer.md` |
| Floating-point constants (`float 3.14`, hex floats) | `03-constants/02-floating-point.md` |
| String constants (`c"...\00"`) | `03-constants/03-strings.md` |
| Global vs local constants, linkage, visibility | `03-constants/04-global-vs-local.md` |
| `alloca` | `04-memory/01-alloca.md` |
| `load`, `store`, atomic load/store | `04-memory/02-load-store.md` |
| Global variables, linkage types, TLS | `04-memory/03-global-variables.md` |
| `addrspace(N)`, `addrspacecast` | `04-memory/04-address-spaces.md` |
| Unconditional `br label %X` | `05-control-flow/01-unconditional-br.md` |
| Conditional `br i1 %c, label %t, label %f` | `05-control-flow/02-conditional-br.md` |
| `switch` | `05-control-flow/03-switch.md` |
| `indirectbr`, `blockaddress` | `05-control-flow/04-indirectbr.md` |
| Pitfalls overview | `08-pitfalls/README.md` |
| Nested instruction-as-expression syntax errors | `08-pitfalls/01-nested-instruction-expressions.md` |
| PHI node predecessor mismatch | `08-pitfalls/02-phi-predecessor-mismatch.md` |
| Duplicate block labels / SSA names | `08-pitfalls/03-duplicate-block-labels.md` |
| Duplicate symbol definitions across modules | `08-pitfalls/04-duplicate-symbols.md` |
| Type schema drift across modules | `08-pitfalls/05-type-schema-drift.md` |
| `immarg` parameter violations on intrinsics | `08-pitfalls/06-immarg-violation.md` |
| Formal Textmapper grammar | `10-grammar/llvm-ir.tm` |
| Grammar notes / how to use it | `10-grammar/README.md` |
| Instruction quick reference | `reference/instruction-quickref.md` |
| Intrinsics list | `reference/intrinsics.md` |
| Glossary | `reference/glossary.md` |

## By instruction (most common)

| Instruction | Read |
|---|---|
| `add`, `sub`, `mul`, `sdiv`, `udiv`, `srem`, `urem` | `01-syntax/02-instruction-format.md`, `reference/instruction-quickref.md` |
| `fadd`, `fsub`, `fmul`, `fdiv`, `frem`, `fneg` | `01-syntax/02-instruction-format.md`, `reference/instruction-quickref.md` |
| `and`, `or`, `xor`, `shl`, `lshr`, `ashr` | `reference/instruction-quickref.md` |
| `alloca` | `04-memory/01-alloca.md` |
| `load`, `store` | `04-memory/02-load-store.md` |
| `getelementptr` (GEP) | `02-types/02-composite-types.md`, `reference/instruction-quickref.md` |
| `br`, `switch`, `indirectbr`, `ret`, `unreachable` | `05-control-flow/` (all four files) |
| `phi` | `00-foundations/02-ssa.md` |
| `icmp`, `fcmp` | `reference/instruction-quickref.md` |
| `select` | `reference/instruction-quickref.md` |
| `call`, `invoke`, `callbr` | `reference/instruction-quickref.md` |
| `atomicrmw`, `cmpxchg`, `fence` | `reference/instruction-quickref.md` |
| `extractvalue`, `insertvalue`, `extractelement`, `insertelement`, `shufflevector` | `reference/instruction-quickref.md` |
| `trunc`, `zext`, `sext`, `fptrunc`, `fpext`, `fptoui`, `fptosi`, `uitofp`, `sitofp` | `reference/instruction-quickref.md` |
| `bitcast`, `addrspacecast`, `inttoptr`, `ptrtoint` | `reference/instruction-quickref.md` |
| `landingpad`, `catchpad`, `cleanuppad`, `catchswitch` | `reference/instruction-quickref.md` |

## By symbol

| Symbol | Means | See |
|---|---|---|
| `%foo`, `%42` | Local (function-scope) identifier | `01-syntax/01-modules-functions-blocks.md` |
| `@foo`, `@42` | Global identifier | `01-syntax/01-modules-functions-blocks.md` |
| `!N`, `!"str"`, `!{...}` | Metadata | `01-syntax/03-comments-metadata.md` |
| `#N` | Attribute group ID | `reference/glossary.md` |
| `$foo` | Comdat name | `reference/glossary.md` |
| `i1`, `i8`, `i32`, `i64`, `iN` | Integer of N bits | `02-types/01-primitive-types.md` |
| `ptr` | Generic pointer (opaque) | `02-types/03-opaque-and-pointer-types.md` |
| `;` | Comment to end of line | `01-syntax/03-comments-metadata.md` |
| `c"..."` | C-style char array constant | `03-constants/03-strings.md` |

## By keyword (where it's introduced)

| Keyword | File |
|---|---|
| `define`, `declare` | `01-syntax/01-modules-functions-blocks.md` |
| `target datalayout`, `target triple` | `01-syntax/01-modules-functions-blocks.md` |
| `source_filename` | `01-syntax/01-modules-functions-blocks.md` |
| `global`, `constant` | `03-constants/04-global-vs-local.md`, `04-memory/03-global-variables.md` |
| `addrspace` | `04-memory/04-address-spaces.md` |
| `thread_local` | `04-memory/03-global-variables.md` |
| `private`, `internal`, `external`, `weak`, `linkonce`, `appending`, `common`, etc. | `04-memory/03-global-variables.md` |
| `default`, `hidden`, `protected` | `04-memory/03-global-variables.md` |
| `dllexport`, `dllimport` | `04-memory/03-global-variables.md` |
| `dso_local`, `dso_preemptable` | `04-memory/03-global-variables.md` |
| `inbounds` | `02-types/02-composite-types.md` |
| `nsw`, `nuw`, `exact`, `fast`, `nnan`, `ninf`, `nsz`, `arcp`, `contract`, `afn`, `reassoc` | `reference/instruction-quickref.md` |
| `align`, `alignstack` | `04-memory/02-load-store.md` |
| `volatile`, `atomic`, `syncscope`, `seq_cst`, `acquire`, `release`, `monotonic`, `unordered`, `acq_rel` | `04-memory/02-load-store.md` |
| `blockaddress`, `indirectbr` | `05-control-flow/04-indirectbr.md` |

## Cross-references to the BCIR project

Real-world examples of LLVM IR concepts (and bugs) live next door:

| Concept | BCIR file | Pitfall |
|---|---|---|
| Cache-line-aligned struct layout | `include/bcir/bcir_ir.hpp` (`BcirClaimV1`) | none |
| LLVM substrate ABI | `runtime/llvm/bcir_master_reference_v2.ll` | metadata-string syntax (fixed in `1f62e86`) |
| Boolean expression construction | `runtime/llvm/bcir_claim_verify.ll` | `08-pitfalls/01-nested-instruction-expressions.md` |
| PHI predecessors | `runtime/llvm/bcir_batch_executor.ll` | `08-pitfalls/02-phi-predecessor-mismatch.md` (fixed in `5754354`) |
| Duplicate block labels | `runtime/llvm/bcir_claim_verify.ll` (pre-`1f62e86`) | `08-pitfalls/03-duplicate-block-labels.md` |
| Cross-module function definition collision | `runtime/llvm/bcir_gem_seed.ll` vs `bcir_worklist.ll` | `08-pitfalls/04-duplicate-symbols.md` |
| Type schema drift | `%bcir.blob.header` in three files | `08-pitfalls/05-type-schema-drift.md` |
| `llvm.prefetch` immarg | `runtime/llvm/bcir_prefetch_profiles.ll` | `08-pitfalls/06-immarg-violation.md` |
