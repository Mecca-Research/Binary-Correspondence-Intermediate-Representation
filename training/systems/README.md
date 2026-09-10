# Systems — languages, build systems, and the kernel boundary

> **Status: PLANNED.** This folder has a scope and no lessons yet. Nothing here
> is teaching material, and nothing in it is gated. See
> [`../ROADMAP.md`](../ROADMAP.md) Phase 3 for when it opens, and
> [`../CORPUS_STANDARD.md`](../CORPUS_STANDARD.md) for what it must deliver.

## Scope

The programs that run closest to the machine and the tooling that builds
them. Where `llvm/` teaches the compiler, this teaches what is compiled
and what it runs on.

### In scope

- C and C++ as systems languages: memory, lifetime, ABI, undefined behaviour
- CUDA and the host/device split
- Build systems: Make, CMake, and what a build graph actually guarantees
- Shell and Bash as programming languages, including their failure modes
- Device drivers, the microkernel boundary, and the Linux kernel
- C# where it meets the systems boundary

### Out of scope

- Language tutorials for their own sake; the angle is always the machine
- Compiler internals, which belong to `llvm/`
- Application frameworks, which belong to `backends/`

## Reference manuals to break down first

A subject folder opens only after its primary references are inventoried, so
the lessons cite normative sources rather than recollection. For this subject:

- ISO/IEC 9899 (C) and ISO/IEC 14882 (C++), at the revisions taught
- The System V and per-target ABI documents
- The CUDA programming guide and PTX ISA reference
- POSIX.1 for the shell and utilities
- The Linux kernel's in-tree documentation, at a pinned release

## Depends on

`hardware/` for the ISA and memory material the ABI lessons rest on.

## Verification plan

- Every example compiles under a pinned toolchain and runs where it can
- Undefined-behaviour lessons carry a sanitizer run showing the diagnosis
- Build-system lessons prove the claimed incrementality, not just describe it
- Kernel material is read-only unless a testable module accompanies it

## Corpus obligations

Like every subject, this folder is complete only at Tier 3: chapters whose
factual claims are gated, chunks that trace to a line, and distillation records
each backed by a gate. Prose alone does not close it.
