# `ptrtoaddr`: the one instruction LLVM 23 added

Between LLVM 18 and LLVM 23 exactly one instruction arrived in the language.
Everything else in that delta is an attribute, an intrinsic, or — as
[chapter 01](01-reading-the-delta.md) shows — not a language change at all.

```llvm
%a = ptrtoaddr ptr %p to i64
```

It looks like a synonym for `ptrtoint`. It is not, and the difference is the reason it
exists.

## What it is

LLVM's own header says it in six words: *a cast from a pointer to an address
(non-capturing `ptrtoint`)*. Two claims live in that line.

**It produces an address, not a pointer's bits.** On most targets those are the same
number. On a target where a pointer carries more than an address — a capability, a fat
pointer, a segmented or tagged pointer — they are not. LLVM models this in the data
layout: an address space declares a pointer size *and* an index (address) size, and they
may differ.

**It does not capture the pointer.** `ptrtoint` is an escape: once a pointer's bits are
an integer, the optimiser must assume the integer could be turned back into a pointer and
used, so it stops reasoning about what the pointer can alias. `ptrtoaddr` asks only
"where does this point?", which is not enough to reconstruct the pointer on a target
where a pointer is more than its address. Taking an address no longer costs the analysis
that taking the bits does.

This is the same shift in thinking that produced the `captures(...)` attribute in the
same era — LLVM moving from *captured / not captured* to a finer statement of **which
part** of a pointer escapes.

## The distinction, made visible

Both casts are 64 bits wide on an ordinary flat address space, so nothing there
distinguishes them. Declare an address space whose pointers are wider than their
addresses and the verifier separates them immediately:

```llvm
; 128-bit pointers whose address is 64 bits
target datalayout = "e-p:64:64-p1:128:128:128:64"

define i64 @address_of(ptr addrspace(1) %p) {
  %a = ptrtoaddr ptr addrspace(1) %p to i64   ; the address: 64 bits
  ret i64 %a
}

define i128 @pointer_bits_of(ptr addrspace(1) %p) {
  %i = ptrtoint ptr addrspace(1) %p to i128   ; the whole pointer: 128 bits
  ret i128 %i
}
```

Ask `ptrtoaddr` for the pointer's full width instead and the verifier refuses:

```
PtrToAddr result must be address width
  %a = ptrtoaddr ptr addrspace(1) %p to i128
```

That message is worth remembering. It is the fastest way to discover that an address
space you assumed was flat is not — the verifier is telling you the pointer carries
something the address does not.

The runnable pair is
[`examples/ptrtoaddr-vs-ptrtoint.ll`](examples/ptrtoaddr-vs-ptrtoint.ll).

## Which one to emit

| You want | Use | Why |
| --- | --- | --- |
| The numeric address, for printing, hashing, alignment checks | `ptrtoaddr` | Says what you mean; keeps alias analysis |
| To round-trip back to a usable pointer | `ptrtoint` | `ptrtoaddr` may have discarded bits |
| To compare two pointers for ordering | `icmp` on the pointers | Neither cast; comparison is defined on pointers |
| To store a pointer in an integer field | `ptrtoint` | You need the pointer, not the address |

The default is **`ptrtoaddr` whenever you only need the address**. It is the more precise
statement, and precision is what the optimiser is able to use.

## Why this file will not assemble on your compiler

`ptrtoaddr` does not exist before LLVM 23. On 18 and 15 the assembler does not report a
type error or a verifier failure — it does not recognise the token at all:

```
llvm-as-18: error: expected instruction opcode
  %a = ptrtoaddr ptr %p to i64
       ^
```

That is why the example carries `; REQUIRES: llvm >= 23` on its first line.
`verify-examples.sh` reads that directive, skips the file on an older assembler with the
reason printed, and verifies it normally once the assembler is new enough — so the corpus
can teach a construct newer than its own baseline without either pretending the baseline
assembles it or dropping the check.

The skip has an owner: CI's `LLVM training corpus (LLVM 23)` job installs 23, so the file
is verified on every run there. A version requirement is a statement about *where* a
check happens, never a way to avoid one.

## Pitfalls checklist

- Do not treat `ptrtoaddr` and `ptrtoint` as spellings of one operation because they
  agree on x86-64. They agree on flat address spaces and nowhere else.
- Do not reach for `ptrtoint` out of habit when you only need an address; you are paying
  for an escape you did not want.
- Do not expect `ptrtoaddr` to round-trip. There is no guarantee the address is enough to
  rebuild the pointer — that is the entire point.
- Do not write `ptrtoaddr` in code that must build on LLVM 18; there is no fallback
  spelling, so the version requirement is real and belongs in your build system.

## Checks

The example is assembled and verified by
[`../tools/verify-examples.sh`](../tools/verify-examples.sh) wherever `llvm-as` is 23 or
newer, and skipped with its reason printed anywhere older.
[`../tools/verify-langref-delta.py`](../tools/verify-langref-delta.py) additionally
requires this chapter to exist and to name `PtrToAddr`: the instruction is marked
`taught` in the disposition table, and a citation that does not mention its own subject
is a finding there.
