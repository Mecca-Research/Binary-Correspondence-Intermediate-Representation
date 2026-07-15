# BCIR C memory discipline and driver/IPC boundary

This document is the implementation policy for BCIR-owned C memory. It is separate
from LangRef R21: R21 describes pointer-lifetime events in a compiled program;
this policy governs allocations made by BCIR's own compiler, model tools, runtimes,
and driver adapters.

## Runtime classes

The machine-checked inventory is
[`runtime/c/MEMORY_CLASSIFICATION.txt`](../../runtime/c/MEMORY_CLASSIFICATION.txt).
Every production `bcir_*.c` translation unit belongs to exactly one class.

| Class | Allocation and ownership contract |
|---|---|
| Freestanding core | No heap or hosted libc. Callers own every buffer and pass its capacity. Integer widths are explicit and failure is deterministic. |
| Hosted tools | Dynamic storage is allowed only through `bcir_host_allocator`. Growth is overflow-checked and two-phase. Public objects have an init/free or context init/reset/destroy contract. |
| Driver adapters | No heap in the direct ABI. Resources cross the boundary as opaque generation-tagged handles and byte offsets, never data pointers. |

`bcir_host_alloc.h` supplies the default libc adapter, checked addition,
multiplication, alignment and geometric growth, two-phase array reallocation, and a
per-operation arena. A custom allocator may fail any allocation attempt without
replacing process-global `malloc`.

## Required ownership rules

- Each allocation has one owner. Public headers label borrowed inputs and owned
  results; ownership transfers only where a function explicitly says so.
- Output objects start in an initialized empty state. Destroy/free operations are
  idempotent. A failed operation leaves no executable or emitted partial artifact.
- `realloc` is always two-phase: failure preserves the original pointer, bytes, and
  capacity.
- Every size addition, multiplication, alignment, and capacity growth is checked
  before allocation or pointer arithmetic.
- Hosted compiler scratch uses per-operation arenas. Returned claim graphs and model
  objects own their storage independently of the parser context.
- Rings are bounded and declare either backpressure or overwrite-oldest behavior.
  Silent overwrite is forbidden.

The preprocessor and C frontend expose re-entrant context APIs. Their original entry
points remain compatibility wrappers over process-static contexts and are explicitly
non-thread-safe. The Q8 loader and Llama inference runtime expose allocator-injected
forms while preserving their libc-default APIs.

## Direct driver ABI first

[`bcir_runtime_channel.h`](../../runtime/c/bcir_runtime_channel.h) is the append-only v1
in-process hook table: open, claim, map, submit, sync/cancel, event delivery, and
close. The value contract uses generation-tagged handles, mappings expressed as byte
offsets and lengths, sequence numbers, and explicit queue policy. The allocation-free
resident loopback driver is the behavioral baseline; it is not a claim that a UART or
other hardware driver has landed. Its submissions complete synchronously, so canceling
an already completed or never-submitted sequence returns `STALE`; asynchronous drivers
may instead deliver a generation-matched `CANCELED` event.

The first hardware driver remains in-process. Its real teardown, cancellation, error
mapping, event, and backpressure needs must stabilize this hook contract before a
transport is implemented.

## IPC boundary

IPC does not enter the freestanding core and is not a leak-prevention mechanism. An
out-of-process Linux adapter is justified only by privilege isolation, crash
containment, vendor-library isolation, or multi-client sharing. When justified, the
initial design is:

- Unix `SOCK_SEQPACKET` control messages;
- bounded `memfd_create`/`mmap` shared rings for bulk data;
- `eventfd` with `epoll` notification;
- offsets and generation-tagged handles, never shared process pointers;
- explicit peer death, close, restart, cancellation, and saturation behavior.

SysV IPC, POSIX message queues, custom futex protocols, and `io_uring` are not initial
dependencies. They require a measured driver need and direct-vtable-versus-transport
behavioral parity.

## Validation cadence

Every `runtime/c` change runs the static class checker, deterministic allocator-failure
sweep, strict warnings for the new safety substrate, ASan/UBSan/LSan, the complete C
runtime gate, differential parity, bounded fuzzing, and native Windows/ARM CI jobs.
The local workstation runs only bounded native tests; ARM execution remains native CI.

The scheduled deep sweep adds longer fuzzing, Valgrind, and available static analyzers.
A manual full bug sweep is required before a release, after allocator or wire-format
changes, and before a new driver family. Every discovered defect receives a
deterministic regression; a sweep report alone is not a gate.
