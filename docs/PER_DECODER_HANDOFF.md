# Plan-driven PER decoder — build reference

A refresh point for the slice that gives phase H's schema-directed decode table its second
row. Written first, deliberately: the work spans four rails and the earlier ECN slices showed
that a finding written down before the code is worth more than a half-finished branch.

## Why this exists

PR #719 built `directed_decode_table` as a **second** table, because the schema-free one could
never hold an OER row — X.696 §6.2 denies OER a schema-free decode permanently, and `CostRow`
needs both axes. That table currently has **one** row (CANONICAL-OER), so "selects OER on
decode latency" is true and *vacuous*: there is nothing to select against.

X.691 §7.2 bars only the schema-**free** decode of PER. A schema-directed one is exactly what
the plan already describes, so this is a **gap in this repository, not a prohibition in the
standard** — and it needs no hardware, which is why it outranks every other phase-H item.

## What already exists

- `runtime/c/bcir_per.h` — the complete bit reader: `reader_init`, `get_bits`, `get_bit`,
  `align`, `bits_for_range`, `constrained`, `semi_constrained`, `unconstrained`,
  `normally_small`, `normally_small_length`, `length`. **No whole-value decode.**
- `runtime/c/bcir_oer.{h,c}` — the model to mirror (161 + 277 lines): a flat `bcir_oer_field[]`
  the caller fills from a plan, and `bcir_oer_decode_sequence` walking it iteratively.
- `bcir/asn1/native_bench.py` — `oer_fields_for(plan)`, `DIRECTED_DECODE_OPS`, and
  `_DIRECTED_REASONS`, where PER's absence is already spelled as a GAP.
- `runtime/c/bcir_asn1_bench.c` — the `dircase` verb with an OER arm to sit beside.

## The three things PER adds over OER

Engineering, not roadblocks. Each has a decided answer:

1. **Presence bitmap preamble (§18.2).** One bit per OPTIONAL/DEFAULT root component, before
   any field. OER's equivalent is octet-aligned and PER's is *bit*-aligned, so it is read with
   `get_bit` in a loop rather than with a padded-octet helper.
2. **ALIGNED vs UNALIGNED are two decoders over one field table.** `bcir_per_align` is the only
   difference at each field boundary — but it is at *every* boundary, so it belongs on the
   **decode call**, not on the field. One `aligned` flag, threaded through.
3. **Extension marker (§18.1).** A leading bit when the type is extensible, and §18.8's
   unknown-extension skip. Held on the decode call for the same reason as `aligned`: it is a
   property of the type, not of any component.

## The wiring trap, already paid for once

`#asn1bench`'s link line in `tools/c/check_runtime.sh` **and** `native_bench._SOURCES` must
both gain the new `.c`. That exact drift broke PR #719 mid-session — the Python side was
updated and the shell gate was not, and it surfaced as an undefined reference in the C gate
rather than anywhere useful. `test_the_gate_and_the_harness_link_the_same_sources` exists to
catch a repeat in the fast tier.

## Segments

| # | Rail | Deliverable |
|---|---|---|
| 0 | docs | this file |
| 1 | C | `bcir_per_plan.{h,c}`: `bcir_per_field`, `bcir_per_decode_sequence` |
| 2 | gates | `#per` gate in `check_runtime.sh` (freestanding, `-Werror`, `-O0 == -O3`) + the `#asn1bench` link line |
| 3 | Python | `per_fields_for(plan)`, `DIRECTED_DECODE_OPS` entries, the harness `dircase` arm |
| 4 | tests/docs | fast-tier tests, roadmap phase H rewrite, `STATUS.md` **last** |

## Verification discipline that earned its keep in the ECN slices

- **Ask whether two inputs differing only in the new thing differ in the digest.** Two silent
  drops were caught this way and by nothing else.
- **Pin exact sets, not membership.** Four table edits became deliberate rather than silent.
- **Read two rails out of their own sources rather than mirroring into a third list.** That
  caught a governor table disagreeing with the oracle on its first run.
- **Exact-match edits, never scripted `str.replace` on a large module.** Three mis-targets came
  from anchors that matched somewhere else or line ranges computed wrong.
- **`STATUS.md` is regenerated LAST**, via `python tools/docs/gen_status.py > docs/STATUS.md`.

## Done means

`directed_decode_table` has rows to choose between, and the roadmap's phase H blockquote —
currently "true and *vacuous*: there is nothing to select against" — is rewritten to say what
the gate now shows.
