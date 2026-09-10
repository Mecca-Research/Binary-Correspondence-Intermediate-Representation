# ABI and Target Lowering

## TL;DR

**The IR signature is not the C signature.** Argument classification happens in
the frontend, per target, before LLVM sees anything. The same C source compiled
for two targets produces two different IR signatures, and both are correct.

The comparison below is generated from one file,
[`examples/abi-boundary.c`](examples/abi-boundary.c), and asserted by
[`../tools/verify-frontend-lowering.py`](../tools/verify-frontend-lowering.py):

| C declaration | x86-64 SysV | AArch64 AAPCS |
| --- | --- | --- |
| `int take_small(struct Small)` (8 B) | `i32 @take_small(i64)` | `i32 @take_small(i64)` |
| `int take_medium(struct Medium)` (12 B) | `i32 @take_medium(i64, i32)` | `i32 @take_medium([2 x i64])` |
| `double take_large(struct Large)` (64 B) | `double @take_large(ptr byval(%struct.Large) align 8)` | `double @take_large(ptr)` |
| `struct Large return_large(void)` | `void @return_large(ptr sret(%struct.Large) align 8)` | same |
| `char narrow(char, short, _Bool)` | `signext i8 @narrow(i8 signext, i16 signext, i1 zeroext)` | `i8 @narrow(i8, i16, i1)` |

Read the table as five separate rules.

## Rule 1: small aggregates are coerced into registers

`struct Small { int a; int b; }` is 8 bytes, so SysV classifies both eightbytes
as INTEGER and passes the whole struct in one general-purpose register. The
frontend expresses that by **changing the parameter type to `i64`** and
reassembling the struct in the callee's entry block.

The struct type still exists in IR (`%struct.Small`), but it is not the
parameter type. That mismatch between "the type the source declared" and "the
type the ABI transports" is the essence of frontend ABI lowering.

## Rule 2: the split point is target law, not size arithmetic

A 12-byte struct is two registers on both targets, but the *spelling* differs:
SysV says `(i64, i32)` — two separate parameters — while AAPCS says
`([2 x i64])` — one array parameter. Those are different IR, different
register assignments, and different rules for what happens when the register
budget runs out.

There is no portable formula. The classification algorithm is written down in
each ABI document (System V AMD64 §3.2.3, AAPCS64 §5.4), and Clang implements
one `TargetInfo`/`ABIInfo` pair per target precisely because they do not
generalize.

## Rule 3: large aggregates go to memory, and `byval` is not universal

SysV passes the 64-byte struct in memory and marks it `byval(%struct.Large)` —
which tells LLVM *the callee receives a copy the caller made*, so the callee may
freely modify it. AArch64 passes a plain pointer, because AAPCS specifies an
indirect parameter whose copy responsibilities differ.

`byval` is therefore an ABI-encoding attribute, not a generic "pass by
reference" marker. Adding or removing it changes who copies what and when.

## Rule 4: large returns become `sret` out-parameters

Both targets agree here:

```llvm
define dso_local void @return_large(ptr dead_on_unwind noalias writable
                                    sret(%struct.Large) align 8 %0)
```

The function's IR return type is `void`; the caller allocates the storage and
passes a pointer. `noalias` says the caller's buffer is not reachable any other
way; `writable`/`dead_on_unwind` narrow the contract further.

Those last two are LLVM 19 attributes, so they appear in the block above (which
is quoted Clang output) but **not** in the checked-in
[`examples/abi-boundary.ll`](examples/abi-boundary.ll) snapshot: that file has to
assemble at this corpus's LLVM 15 baseline, and the normalizer strips parameter
attributes newer than the baseline. The `sret` contract the section is about is
unaffected.

`sret` must appear on the declaration **and** every call site. This is the most
common hand-written-IR ABI break: a declaration that agrees with the ABI and a
call site that does not still verifies, and then passes arguments one slot off.

## Rule 5: narrow integer extension is the caller's job — on some targets

```llvm
; x86-64 SysV
define dso_local signext i8 @narrow(i8 noundef signext %0, i16 noundef signext %1,
                                    i1 noundef zeroext %2)
; AArch64
define dso_local i8 @narrow(i8 noundef %0, i16 noundef %1, i1 noundef %2)
```

SysV requires values narrower than the register to arrive extended, and Clang
records that with `signext`/`zeroext` on both the parameters and the return.
AAPCS does not, so the attributes are absent.

Note `_Bool` is `zeroext` while `char`/`short` are `signext`: extension follows
the *source* signedness, and `_Bool` has only two valid values, so any bit
pattern other than 0 or 1 in the register is UB.

Dropping `signext` when hand-writing or rewriting IR produces a callee that
reads garbage in the upper bits — on the targets that require it, and only
there, which makes it a bug that reproduces on one CI cell.

## Varargs

The frontend, not LLVM, emits the `va_list` machinery. On SysV a `va_list` is a
`%struct.__va_list_tag` with a register-save area and an overflow area; on
AArch64 it is a different struct; on some targets it is just a pointer. The
`llvm.va_start`/`va_arg`/`va_end` intrinsics mark the boundaries, but the
per-argument fetch code is target-specific and frontend-generated.

Also: the variadic call site must spell the full function type,
`call i32 (ptr, ...) @printf(...)` — see
[`../05-control-flow/05-call-and-ret.md`](../05-control-flow/05-call-and-ret.md).

## What `-target` actually changes

| Category | Examples |
| --- | --- |
| Type sizes and alignment | `long`, `long double`, pointer width, `max_align_t` |
| Default signedness | `char` is signed on x86-64 Linux, unsigned on AArch64 Linux |
| Data layout string | endianness, alignment, address-space sizes, stack alignment |
| Argument classification | everything in the table above |
| Available builtins/intrinsics | target-specific `__builtin_*` |
| Predefined macros | `__x86_64__`, `__aarch64__`, `__SIZEOF_LONG__`, ... |

`-target` is a semantic switch. Cross-compiling with the wrong triple produces
a program that compiles and is wrong in ways no IR-level review will catch.

## Inspecting ABI decisions

```bash
clang -S -emit-llvm -target x86_64-unknown-linux-gnu abi-boundary.c -o - | grep '^define'
clang -S -emit-llvm -target aarch64-unknown-linux-gnu abi-boundary.c -o - | grep '^define'
clang -Xclang -fdump-record-layouts -c file.c -o /dev/null   # field offsets
clang -dM -E -target aarch64-unknown-linux-gnu - </dev/null  # predefined macros
llvm-readobj --elf-output-style=GNU -s object.o              # what actually shipped
```

Diffing the two `grep '^define'` outputs is the fastest ABI review that exists,
and it is the mechanism this chapter's gate uses.

## Pitfalls

- **Assuming an IR signature is portable.** It is per-triple.
- **Hand-writing a declaration that "looks like" the C one.** Match what Clang
  emits for the same target, or use Clang to emit it.
- **Putting `byval`/`sret` on only one side.** Verifies, then breaks.
- **Forgetting `signext`/`zeroext`.** Fails only on targets that need it.
- **Assuming `byval` means "the callee gets a reference".** It means the callee
  gets a copy the caller made.
- **Assuming `char` signedness.** Target-dependent, and it changes comparisons.
- **Treating struct return as free.** `sret` adds an argument, a caller-side
  allocation, and an aliasing contract.
- **Using `-march=native` in a build meant to be portable.** It is a target
  change, and it will produce instructions the deployment host may not have.

## BCIR notes

- BCIR's C-front rail carries a **target ABI matrix** for exactly the reason
  this page documents: type layout and argument classification are the part of
  "compile this C" that cannot be inferred from the source. See
  [`../../docs/languages/CFRONT_GUIDE.md`](../../docs/languages/CFRONT_GUIDE.md).
- Every frozen BCIR ABI — StreamPack, BCAB, the telemetry frame — is specified
  in **explicit octets with explicit alignment**, not in C struct declarations,
  so that the wire format does not silently change when a target does. That is
  the same lesson this page teaches, applied one level up.
- When a BCIR runtime call is emitted into LLVM IR, the attribute set on the
  declaration and on the call site must be produced by the same code path.
  Emitting them from two places is how the one-rail drift in the repository's
  own history happened.

## See also

- [`03-c-lowering-rules.md`](03-c-lowering-rules.md) — aggregate layout
- [`../05-control-flow/05-call-and-ret.md`](../05-control-flow/05-call-and-ret.md) — call sites and ABI attributes
- [`../13-advanced-ir/04-attributes.md`](../13-advanced-ir/04-attributes.md) — the attribute catalog
- [`../12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md) — what the backend does with these signatures
