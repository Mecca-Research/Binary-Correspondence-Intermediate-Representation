# Index: Concurrency, atomics, and memory ordering

Use this index when the program is correct on one thread and wrong on several,
or when a variable that "should" be safe is not.

| Concept | Where it is covered |
|---|---|
| Choosing a memory ordering for an atomic operation | [`08-pitfalls/09-atomic-ordering-mismatch.md`](../08-pitfalls/09-atomic-ordering-mismatch.md) |
| An ordering too weak to hold the invariant it guards | [`08-pitfalls/09-atomic-ordering-mismatch.md`](../08-pitfalls/09-atomic-ordering-mismatch.md) |
| Why `volatile` does not make a race safe | [`08-pitfalls/10-volatile-is-not-atomic.md`](../08-pitfalls/10-volatile-is-not-atomic.md) |
| Telling device access apart from thread synchronization | [`08-pitfalls/10-volatile-is-not-atomic.md`](../08-pitfalls/10-volatile-is-not-atomic.md) |
