# Hardware Counters, and Refusing Honestly

## The failure this chapter exists to prevent

A counter harness runs. Every counter comes back `0`. `0` is written into a
report. A cache-miss rate of zero becomes a finding, an IPC of infinity becomes
a highlight, and nobody notices, because a zero looks exactly like a
measurement.

The fix is structural, not procedural: **an unreadable counter must have a
representation that is not a number.**

```json
"instructions": null,          // correct
"instructions": 0,             // a lie with the same shape as data
```

## Privilege is not capability

The instinct on a failed counter read is to reach for permissions:
`perf_event_paranoid`, `CAP_PERFMON`, `sudo`. That answers the wrong question.
`perf_event_paranoid` describes *who may* open a counter. It says nothing about
whether the counter *exists*.

On a virtualized host the PMU is commonly not exposed to the guest at all, and
the syscall fails regardless of privilege — with `ENOENT`, or with `EACCES` when
the paranoid level rejects the request before the kernel looks. Run
[`../tools/probe-hardware-counters.py`](../tools/probe-hardware-counters.py) and
read the answer, rather than inferring capability from permission. Observed on
the container this chapter was written in:

```
euid          : 0
paranoid level: 2
PMU devices   : breakpoint, msr, power, software, tracepoint, uprobe
perf binary   : (absent)
virtualization: hypervisor flag present in /proc/cpuinfo

counter                   value  refusal / note
cpu_cycles                 null  ENOENT: No such file or directory
instructions               null  ENOENT: No such file or directory
cache_references           null  ENOENT: No such file or directory
cache_misses               null  ENOENT: No such file or directory
branch_instructions        null  ENOENT: No such file or directory
branch_misses              null  ENOENT: No such file or directory

STATUS: unavailable
```

Root, and no hardware counters. Note the PMU device list: `breakpoint`, `msr`,
`power`, `software`, `tracepoint`, `uprobe` — every one of them synthetic. There
is no `cpu` entry, which is the actual hardware PMU, and its absence is the
whole answer. Raising privilege from that state changes nothing.

## What the probe checks, and in what order

1. **The syscall itself.** `perf_event_open` for each hardware event. This is
   the only authoritative test, and it is the one the probe performs. Everything
   else is context for interpreting its answer.
2. **`perf_event_paranoid`.** Context for reading the errno — and note that at
   level 3 or higher it *masks* the distinction, because permission is checked
   before availability.
3. **`/sys/bus/event_source/devices`.** Context: whether a hardware `cpu` PMU is
   exposed at all.
4. **Virtualization hints.** Context: a hypervisor flag makes `ENOENT` expected
   rather than surprising.

The errno is the diagnosis:

| errno | Means | Action |
| --- | --- | --- |
| `ENOENT` | the event does not exist on this host | no PMU; do not retry with privilege |
| `EACCES` / `EPERM` | you may not open it — *and see below* | lower `perf_event_paranoid`, or grant `CAP_PERFMON` |
| `ENOSYS` | no `perf_event_open` at all | not a Linux perf host |
| `EINVAL` | malformed attribute, or an unsupported combination | fix the request |
| `EMFILE` / `ENOSPC` | out of descriptors or counter slots | multiplex, or ask for fewer events |

**`EACCES` does not prove the counter exists.** At `perf_event_paranoid` 3 or
higher the permission check happens *before* the kernel decides whether the
event is available, so an absent PMU and a forbidden one answer identically.
That is why the probe records the PMU device list next to the errno: if there
is no `cpu` entry there, raising privilege will convert `EACCES` into `ENOENT`
and nothing more.

Two hosts, same verdict, different reason — both real, and neither one has a
hardware PMU:

| | this repository's dev container | a GitHub-hosted CI runner |
| --- | --- | --- |
| euid | 0 | 1001 |
| `perf_event_paranoid` | 2 | 4 |
| PMU devices | breakpoint, msr, power, software, tracepoint, uprobe | breakpoint, kprobe, msr, software, tracepoint, uprobe |
| errno | `ENOENT` | `EACCES` |
| `perf` binary | absent | `/usr/bin/perf` |
| STATUS | `unavailable` | `unavailable` |

The CI runner is the more instructive of the two: it *has* the `perf` binary,
which is exactly the thing that makes people assume counters are available. The
binary is not the capability.

## Counter multiplexing, and why raw counts can be fractional

Hardware has a small number of physical counter slots. Request more events than
there are slots and the kernel time-multiplexes them, running each for part of
the interval and scaling the result up. Two consequences:

- **Scaled counts are estimates.** `time_enabled` and `time_running` in the read
  format tell you the scaling factor. If `time_running < time_enabled`, the
  value was extrapolated — record the ratio with the value, or the number
  silently claims a precision it does not have.
- **Ratios between multiplexed events are unreliable.** Two events scaled from
  different sub-intervals may not have been measured over the same instructions
  at all. Group events that must be compared into one `perf_event_group` so
  they are scheduled together.

## Designing a harness that cannot lie

| Requirement | Why |
| --- | --- |
| Every counter defaults to `null`, and only a successful read replaces it | the unreadable case is representable |
| Record the errno, not just failure | `ENOENT` and `EACCES` need different responses |
| Record `time_enabled` / `time_running` with every value | multiplexing is invisible otherwise |
| Record host, PMU list, paranoid level, virtualization | the same numbers mean different things on different hosts |
| Emit a `status` field with an explicit `unavailable` state | so a consumer can branch on it |
| Provide `--require-counters` for rigs that must have them | the same harness serves both, and says which it was |
| Never substitute a modelled number for a missing measurement | if you must model, label the row `modelled` |

The last row is the one to hold hardest. A harness that fills gaps with model
output produces a table whose rows cannot be told apart.

## The evidence record

The probe emits a JSON document with `counters` (values or `null`), `refusals`
(why each is `null`), `capability` (paranoid level, PMU devices, virtualization
hint), and `status`.

```bash
python3 training/llvm/tools/probe-hardware-counters.py \
  --format json --output build/counter-capability.json
```

Exit status is `0` when the probe completed, **whether or not counters were
available** — "no counters here" is a successful measurement of the host, and a
capability probe that fails when it discovers an incapability cannot be run
where its answer matters most. Pass `--require-counters` on a rig that is
supposed to have them, and it exits nonzero instead.

Store this record next to the measurements it accompanies. A counter table with
no capability record beside it cannot be interpreted a month later.

## What this closes, and what it does not

This is a **capability probe and an evidence-schema harness**. It answers "can
this host read hardware counters, and what exactly did it say", and it makes the
unreadable case unfalsifiable.

It is **not** a full collection harness. Attributing counts to a workload needs
event grouping, per-thread or per-cgroup attribution, multiplexing correction,
and the ring-buffer machinery for sampled events. Building that against a host
with no PMU would produce code nobody could test — and untestable code in a
corpus about evidence would be the wrong lesson. That remains open, and it is
open on **hardware access**, not on effort: see
[`../../docs/BCIR_TARGET_ACCESS.md`](../../../docs/BCIR_TARGET_ACCESS.md).

## Pitfalls

- **Recording an unreadable counter as `0`.** The defining mistake.
- **Reaching for `sudo` on `ENOENT`.** Wrong diagnosis, wasted afternoon.
- **Ignoring multiplexing.** Scaled estimates presented as counts.
- **Comparing counters across hosts.** Event definitions and microarchitectures
  differ; "cache-misses" is not one thing.
- **Using derived rates without the raw counts.** IPC without instructions and
  cycles cannot be checked.
- **Trusting `perf stat` output without reading its scaling column.** It tells
  you when it multiplexed; the column is easy to skip.
- **A probe that exits nonzero on an incapable host.** It then cannot be run
  where its answer is most needed.

## BCIR notes

- This repository records the same finding as a first-class constraint: no
  silicon certificate from a virtualized host, `perf_event_open` returns
  `ENOENT` regardless of privilege, and calibration refuses shared runners for
  frozen tables. The refusal is the feature.
- Two roadmap items here are blocked on *access*, not on code. Saying so
  precisely — naming the syscall, the errno, and the missing PMU — is what makes
  "blocked" a checkable statement rather than an excuse.
- Record an unreadable counter as null, never zero; widen a timing interval to
  the clock quantum. Both rules exist because both mistakes were made once.

## See also

- [`02-measurement-and-profile-collection.md`](02-measurement-and-profile-collection.md) — clocks, profiles, and their biases
- [`05-reporting-and-claim-discipline.md`](05-reporting-and-claim-discipline.md) — measured versus modelled
- [`../15-binary-analysis/02-dynamic-traces-and-counters.md`](../15-binary-analysis/02-dynamic-traces-and-counters.md) — the counter evidence schema
- [`../../docs/kernel/HARDWARE_VALIDATION.md`](../../../docs/kernel/HARDWARE_VALIDATION.md) — what is and is not validated here
