# Workload Selection

## The question a workload answers

A benchmark does not measure "performance". It measures one program, on one
input, on one machine, in one state. Every generalization beyond that is an
argument you have to make, and the workload is where you either earn it or
lose it.

Before writing any code, write down the answer to: **what decision will this
measurement change?** If no decision depends on it, the measurement's only
possible outcome is a number in a document.

## The four failure modes

### 1. The microbenchmark that measures the compiler, not the code

```c
// Times ~0 ns, forever, on every compiler.
for (int i = 0; i < N; ++i)
  result = expensive(input);
```

If `expensive` is pure and `result` is unused, the loop is deleted. If `input`
is a constant, the call is folded. Neither outcome is a performance property of
`expensive`.

Defences, in order of preference:

- Consume the result in a way the compiler cannot see through — accumulate it
  into a value the program returns, or write it through a `volatile` pointer.
- Make the input opaque: read it from a file, take it from `argv`, or pass it
  through a function the optimizer cannot inline.
- Use the compiler barrier your benchmark framework provides (`DoNotOptimize` /
  `ClobberMemory` in Google Benchmark, or an empty `asm volatile` with the
  value in a register constraint).
- Check the emitted code. `clang -S` and a look at the loop body is faster than
  arguing about it. See [`../20-clang-frontend/README.md`](../20-clang-frontend/README.md).

### 2. The workload that is all warm-up

A benchmark whose working set fits in L1 after the first iteration measures L1.
A benchmark that allocates once and reuses the buffer measures neither the
allocator nor the fault path. Whether that is right depends entirely on the
decision being made — and the mistake is not choosing one, it is not *saying*
which one you chose.

State the cache and allocation regime explicitly:

| Regime | Measures | Appropriate when |
| --- | --- | --- |
| Hot, resident working set | steady-state compute | the real workload is a long-running loop |
| Cold caches, fresh pages | first-touch cost, page faults | the real workload runs once per request |
| Streaming, larger than LLC | memory bandwidth | the real workload is data-bound |

### 3. The input that flatters one implementation

A sorting benchmark on already-sorted data, a hash table benchmark with no
collisions, a branch benchmark whose branch is perfectly predicted: each is a
real measurement of an unrepresentative case. The result generalizes to exactly
the inputs it used.

Pick inputs the way a fuzzer would pick them: adversarially, and including the
shapes the implementation would rather not see. If a change wins only on one
input distribution, that is a finding about the distribution, and it belongs in
the report.

### 4. The benchmark that cannot fail

If the candidate is faster on every input you thought to try, you have probably
not tried the input where it is slower. A performance change that is
Pareto-superior is rare; changes usually trade something. Look for the trade
before someone else finds it in production.

## Choosing the size

Two competing constraints:

- **Long enough that the timer's quantum is irrelevant.** If the clock ticks in
  50 ns and the workload runs in 200 ns, a quarter of your measurement is
  quantization. Aim for at least a thousand ticks per sample, and widen the
  interval rather than trusting a high-resolution clock's advertised precision.
- **Short enough to repeat many times.** Statistical power comes from the number
  of samples, not the length of each one. Thirty samples of one second beat one
  sample of thirty seconds, by a wide margin — the single long run has no
  dispersion estimate at all.

When the operation is genuinely shorter than the clock quantum, batch it: run
it `K` times inside one timed region and divide, and report `K` as part of the
method. Do not silently divide.

## The workload set

One workload is an anecdote. A defensible claim needs a *set*, chosen before
the measurement:

- **A representative case** — what the change is meant to improve.
- **A pessimal case** — where the change should hurt most, if it hurts.
- **A neutral case** — untouched by the change, and therefore a check on the
  rig. If the "neutral" workload moves, the rig moved.

The neutral workload is the one people skip and the one that catches the most
mistakes. A methodology that reports a 6% win on the target workload and a 5%
"win" on a workload the change cannot possibly affect has measured the machine.

## What to record with every workload

Recording this is cheap at measurement time and impossible afterwards:

| Field | Why |
| --- | --- |
| Exact command line and working directory | reproduction |
| Compiler, version, and every flag | `-march=native` changes the target |
| Input identity and size (a hash, not a description) | the input *is* the workload |
| Iteration/batch count | so a per-operation number can be un-divided |
| Cache/allocation regime | decides what the number means |
| Host, kernel, CPU model, governor | see [`03-repeated-trials-and-noise-control.md`](03-repeated-trials-and-noise-control.md) |
| Whether the host is virtualized or shared | often the dominant term |

## Pitfalls

- **A dead-code-eliminated benchmark.** Reads as an enormous speedup.
- **Timing the first iteration.** Measures page faults, JIT warm-up, and lazy
  binding.
- **Comparing across machines.** A number from one host and a number from
  another are not a comparison.
- **Reusing a workload past its question.** A microbenchmark chosen to isolate
  one loop does not answer an end-to-end question, however often it is quoted
  as if it did.
- **Changing the workload and the implementation together.** Two variables, one
  measurement, no conclusion.
- **Reporting per-operation cost without the batch factor.** Un-dividing is
  impossible later.

## BCIR notes

- BCIR's cost model is a *plan-time* quantity, and a runtime measurement is a
  different object. Comparing them is legitimate and useful — it is how a model
  is calibrated — but a modelled number must never be reported as a measured
  one, and this repository labels every row accordingly.
- The GEM+ measurement discipline names three metric classes: `exact`
  (deterministic — gates anywhere), `ratio` (a timed ratio, wide band), and
  `wall` (absolute milliseconds — indicative only, never gates). Choose the
  class when you choose the workload, not when you write the report.
- An optimality claim needs a lower bound. Without one, the honest label is
  "best measured", not "optimal" — the TMSAO ladder in
  [`../../docs/research/BCIR_GEMPLUS_ROADMAP.md`](../../../docs/research/BCIR_GEMPLUS_ROADMAP.md).

## See also

- [`02-measurement-and-profile-collection.md`](02-measurement-and-profile-collection.md) — getting the numbers out
- [`../07-optimization/10-lto-bolt-experiment-matrix.md`](../07-optimization/10-lto-bolt-experiment-matrix.md) — a deliberately artifact-only comparison, and why
