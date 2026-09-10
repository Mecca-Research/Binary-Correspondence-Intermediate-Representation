# Measurement and Profile Collection

## Clocks

| Clock | Use | Do not use for |
| --- | --- | --- |
| `CLOCK_MONOTONIC_RAW` | elapsed time, unaffected by NTP slew | wall-clock timestamps |
| `CLOCK_MONOTONIC` | elapsed time, adjusted | anything needing raw ticks |
| `CLOCK_PROCESS_CPUTIME_ID` | CPU time of this process | anything with blocking I/O |
| `CLOCK_REALTIME` | timestamps for logs | intervals — it can go backwards |
| `rdtsc` / `cntvct_el0` | very short intervals | anything, without knowing the frequency and the serialization rules |

Two properties matter and are frequently confused:

- **Resolution** is the smallest difference the clock can express.
- **Quantum** is the smallest difference it actually reports.

`clock_getres` reports the first. The second is measured, by timing an empty
interval repeatedly and looking at the smallest nonzero difference. On some
platforms the gap is dramatic: a clock advertising nanosecond resolution can
advance in steps of tens of nanoseconds, which silently quantizes every
measurement shorter than about a microsecond.

**Widen the interval to the quantum.** If the operation is shorter than a few
hundred quanta, batch it (see
[`01-workload-selection.md`](01-workload-selection.md)) rather than trusting the
clock to resolve it.

## What to time

```c
// Wrong: the timer call is inside the measured region, once per iteration.
for (int i = 0; i < N; ++i) {
  t0 = now(); work(); t1 = now();
  total += t1 - t0;
}

// Right: one timed region around a batch, plus the batch factor recorded.
t0 = now();
for (int i = 0; i < N; ++i) work();
t1 = now();
sample = (t1 - t0);   // report N alongside it, do not silently divide
```

Reading a clock is not free — it can cost tens to hundreds of nanoseconds, and
on some configurations it is a syscall. A per-iteration timer on a
hundred-nanosecond operation measures the timer.

## Sampling profilers

A sampling profiler interrupts the program periodically and records where it
was. That gives you attribution, not timing, and it comes with known biases:

- **Skid.** The recorded instruction pointer is not exactly where the event
  occurred; costs land on nearby instructions. Attribute to a *region*, not a
  line, unless the hardware provides precise events (PEBS / SPE / IBS).
- **Blind spots.** Samples only arrive where interrupts are delivered. Code
  running with interrupts masked, or in a very short-lived thread, is
  under-represented.
- **The inlining problem.** Without frame pointers or accurate unwind data, the
  call stack is wrong, and the cost lands on the wrong caller. Build with
  `-fno-omit-frame-pointer` for profiling builds, or ensure `.eh_frame`/
  `.debug_frame` is present and the profiler uses it.
- **Sampling frequency interacts with periodicity.** A workload whose phases
  align with the sampling period is systematically mis-attributed. Vary the
  frequency and check that the profile is stable.

A profile tells you *where* the time went in the build you profiled. It does
not tell you that removing that cost makes the program faster — the removed
work may be overlapped with something else. That claim needs the A/B
measurement in [`04-significance-and-effect-size.md`](04-significance-and-effect-size.md).

## Instrumentation profilers

Instrumentation counts exactly, and changes the thing it counts. Per-function
entry/exit hooks can dominate small functions, inhibit inlining, and change
code layout enough to move the answer. Use instrumentation for **counts**
(how many times, how many bytes) and sampling for **time**.

`-fprofile-instr-generate` for PGO is a special case: its overhead is
acceptable because its output feeds a compiler, not a report. Do not quote
timings taken from an instrumented build.

## PGO, LTO, and post-link

For the pipeline itself, see
[`../07-optimization/06-pgo-lto-bolt.md`](../07-optimization/06-pgo-lto-bolt.md).
For methodology, three rules:

1. **Profile the workload you will run.** A profile from workload A applied to
   workload B is a plausible-looking source of regressions.
2. **Keep the profile with the build.** A build's performance is a property of
   `(source, flags, profile)`. Reporting a number without naming the profile is
   an incomplete claim.
3. **Do not compare an instrumented build against an optimized one.** They are
   different programs.

## Optimization remarks: measurement-free evidence

```bash
clang -O2 -Rpass=inline -Rpass-missed=loop-vectorize -c file.c
clang -O2 -fsave-optimization-record -c file.c    # YAML, one record per decision
```

Remarks tell you what the optimizer decided and, for missed ones, why. This is
deterministic evidence: it needs no repetitions, no quiet machine, and no
statistics. When the question is "did the vectorizer fire?", a remark answers
it exactly, and a timing run answers it probabilistically.

**Prefer a deterministic observation to a timed one wherever the question
allows it.** Instruction counts, artifact sizes, pass decisions, and emitted
symbol sets are all `exact`-class evidence. Reach for a stopwatch only when the
question is genuinely about elapsed time.

## Recording samples

The analysis tool in this chapter reads a simple JSON document; the schema is in
[`examples/README.md`](examples/README.md). The parts that matter for
collection:

- **One number per sample, in the original run order.** Order is data: it is
  what makes drift detectable. Never write out a sorted series.
- **No pre-aggregation.** Do not record a mean and a standard deviation instead
  of the samples; every robust statistic needs the raw values.
- **No trimming at collection time.** An outlier is evidence until something
  explains it. Record it, then decide.
- **Record the unit and the metric class** (`exact`, `ratio`, `wall`) with the
  data, so a later reader cannot mistake an indicative row for a gating one.

## Pitfalls

- **Timing inside the loop.** Measures the clock.
- **Using `CLOCK_REALTIME` for intervals.** It can step backwards.
- **Trusting advertised clock resolution.** Measure the quantum.
- **Quoting timings from an instrumented or sanitizer build.** Different program.
- **Profiling without frame pointers or unwind info.** Wrong attribution, confidently displayed.
- **Sorting samples before storing them.** Destroys the drift signal permanently.
- **Storing summary statistics instead of samples.** Nothing robust can be recomputed.
- **Using a stopwatch for a question a remark answers.** Slower, noisier, and less conclusive.

## BCIR notes

- The repository's own timing lesson is recorded in
  [`../../docs/BCIR_TARGET_ACCESS.md`](../../docs/BCIR_TARGET_ACCESS.md): widen
  the measured interval to the clock quantum, and record an unreadable counter
  as null rather than zero.
- `exact` rows — deterministic counts, digests, artifact bytes — are preferred
  precisely because they can gate anywhere, on any host, without a quiet
  machine. Most of what this repository gates on is `exact` by design.

## See also

- [`03-repeated-trials-and-noise-control.md`](03-repeated-trials-and-noise-control.md) — how many, and in what order
- [`06-hardware-counter-harness.md`](06-hardware-counter-harness.md) — counters, and refusing honestly
- [`../15-binary-analysis/02-dynamic-traces-and-counters.md`](../15-binary-analysis/02-dynamic-traces-and-counters.md) — evidence schemas
