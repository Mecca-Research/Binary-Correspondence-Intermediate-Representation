# Security Policy

BCIR is a compiler/intermediate-representation research project: an executable Python oracle
(`bcir/`), an MLIR/C++ dialect + passes (`mlir/`), and a **freestanding, no-libc C runtime**
(`runtime/c/`). The runtime decodes an untrusted binary wire format (the StreamPack ABI and the
ETL binary records), so memory-safety of those decoders is the primary security surface.

## Supported versions

This is pre-1.0 software; only the latest `main` is supported. Fixes land on `main` — please
reproduce against a recent commit before reporting.

## Reporting a vulnerability

**Do not open a public issue for a security problem.** Report it privately through GitHub:

1. Go to the repository's **Security** tab → **Report a vulnerability** (GitHub Private
   Vulnerability Reporting), or
2. open a private security advisory at
   `https://github.com/Mecca-Research/Binary-Correspondence-Intermediate-Representation/security/advisories/new`.

Please include: the affected component (oracle / MLIR pass / C runtime), a minimal reproducer
(a `.bcir`/StreamPack input, IR, or test), the observed impact (crash, OOB read/write, UB), and
the commit hash. We aim to acknowledge within a few business days.

## Scope

In scope:

- Memory-safety defects in the C runtime decoders (`runtime/c/`) — out-of-bounds reads/writes,
  integer overflow leading to OOB, use of uninitialized memory — reachable from a crafted
  StreamPack / ETL binary input. The decoders are fuzzed (libFuzzer + ASan/UBSan); a crashing
  corpus entry is a valid report.
- Verifier soundness gaps that let a malformed module pass `-bcir-verify` and then miscompile or
  crash a downstream pass (a violation of laws R1–R18).
- Unsafe handling of inputs in the lowering/codegen path that compiles+runs emitted C/LLVM IR.

Out of scope:

- Denial of service from intentionally pathological but well-formed inputs (the optimizer is a
  research vehicle; unbounded problem sizes are expected to be slow).
- Issues that require modifying the trusted build/toolchain or running attacker-controlled code
  by design (e.g. the AOT/JIT test harness compiling a kernel you supplied).

## Disclosure

We follow coordinated disclosure: we will work with you on a fix and a disclosure timeline, and
credit you in the advisory unless you prefer to remain anonymous.
