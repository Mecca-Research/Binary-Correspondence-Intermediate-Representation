# Solution 025: Attribute contract review

## Attribute classification

- `nonnull` on an argument is primarily a caller-side promise: every dynamic call
  must pass a non-null pointer. On a return value, it is a callee-side promise.
- `dereferenceable(4)` says the pointer can be safely loaded for at least four
  bytes for the duration required by the operation. It implies stronger memory
  validity than a pointer type alone.
- `readonly` on an argument says memory reachable through that argument is not
  written by the function. On a function declaration, it describes the memory
  effects of the callee as a whole.
- `nocapture` says the callee does not retain the pointer beyond the call.
- `noalias` on a returned pointer promises the returned object is not aliased by
  other live pointers in the relevant LLVM sense; it should not be added merely
  because a lookup appears to produce a unique logical resource.

## Review outcome

Keep `readonly` and `nocapture` only when the lowered implementation or source
ABI proves those facts. Keep `nonnull` / `dereferenceable(4)` only when callers
are required to provide valid storage. Keep return `noalias` only when ownership
or allocation rules prove uniqueness. If the BCIR lowering only knows that a
pointer is a handle into a table, it should avoid inventing these attributes and
instead preserve weaker metadata or explicit runtime checks.
