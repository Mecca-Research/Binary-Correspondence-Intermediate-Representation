# BCIR security audit, 2026-08-12

Independent audit of `origin/main` at `3decf69`, started from the findings in
[`BCIR_TMSAO_ASN1_JSON_DRIVER_PROPOSAL.md`](BCIR_TMSAO_ASN1_JSON_DRIVER_PROPOSAL.md) and the
accompanying Codex report, then carried past them. Every claim below was reproduced against
the tree before it was fixed; nothing here is inherited on trust.

**Result: 15 defects confirmed, 14 fixed, 1 deliberately half-closed and tracked.** Two
findings in the source report were investigated and are recorded as *not reproduced*.

---

## 1. The two failure classes

The findings sort into exactly two shapes, and neither is "the decoder crashed". That is
worth stating first, because it changes what a reader should look for next.

**Class A — a second spelling.** A strict decoder accepted a byte string for a value that
the canonical rules already spell exactly once. The decode is *correct*: the right abstract
value comes back. What is lost is uniqueness. BCIR digests the octets it receives and
compares artifacts byte for byte, so one artifact with two accepted encodings is two digests,
and which one you get is chosen by whoever sent the bytes.

**Class B — a vacuous check.** A law returned clean over something it never examined. An
empty StreamPack satisfied the whole R10 provenance law because every loop in it iterated the
pack's segments. This class is the more dangerous of the two in this repository: a certificate
is issued either way, and the certificate is the product.

A third observation cuts across both and is the single most reusable thing in this audit:

> **Twice, a wire format was parsed with a host-language parser.** X.690 §11.3.2's ISO 6093
> field went to Python's `float()`; X.692 §24.7's `INT-TO-CHARS` inverse went to `int()`.
> Both accept a strictly larger language than the clause — PEP 515 underscores, the words
> `inf` and `NaN`, and every Unicode decimal digit — so `1_0` decoded as 10.0 and Arabic-Indic
> `۴۲` as 42. JER already had this right: its scanner enforces the JSON number grammar
> *before* conversion. The fix made the other two follow that pattern.

Any remaining place where untrusted text reaches a host parser without a grammar check first
is a candidate for the same defect. That is a cheap, mechanical sweep, and it is the
recommended follow-up.

---

## 2. Class A — canonical-byte defects

All fixed in `de6c14f`. Each fix asserts both halves: the bad spelling is refused **and** the
encoder's own output still round-trips — a canonicality check that rejects valid encodings
would be a worse defect than the one it replaces.

| # | Clause | What was accepted | Impact |
|---|---|---|---|
| A1 | X.690 §8.1.2 | The requested type's tag was never compared to the wire's. An INTEGER schema returned `True` for `01 01 ff`; a SEQUENCE schema accepted a SET or a constructed `[0]` over identical contents. | The octets choose the type. A pack's root SEQUENCE tag could be swapped and still decode. |
| A2 | X.690 §11.5 | A component present and equal to its DEFAULT, under **strict DER**. The encoder always omitted it. | Two encodings of one pack. |
| A3 | X.696 §31.9 | The same, under `OerRules.CANONICAL`. | Same, on the OER rail. |
| A4 | X.690 §11.3.1 | DER REAL with base 8 or 16 (a §8.5.4 *sender's option*), an even mantissa, a non-zero scaling factor, or a zero mantissa. | `80 01 02` and `80 02 01` are both 4.0 — unboundedly many encodings per value. |
| A5 | X.690 §11.3.2 | The ISO 6093 field parsed by `float()`: `1_0`, `inf`, `NaN`, `1e5`, surrounding whitespace. The NR form selector was read and never enforced, so NR1 — ISO 6093's *integer* form — accepted `1.5e3`. | `inf` and `NaN` already have one-octet §8.5.9 encodings, so these are second spellings. `1_0` is not a number in ISO 6093 at all. |
| A6 | X.690 §8.7.3.2 | Any child tag inside a constructed string; contents concatenated regardless. | **Content laundering.** A constructed VisibleString holding an INTEGER produced a string from the INTEGER's contents octet — DEL included — which `to_der` then re-emitted as valid primitive DER. Arbitrary octets enter any string type, past its repertoire check. |
| A7 | X.697 §4.3 | `elements=0` admitted `[0]`; `members=0` admitted `{"k":0}`. The scanner counted *separators* and only tested the ceiling inside the comma branch. | A resource limit that does not hold at its own boundary, on the one path whose job is bounding untrusted input. |
| A8 | X.692 §24.7.4 | `INT-TO-CHARS` inverted with `int()`: `4_2` → 42, `۴۲` → 42, `４２` → 42. | Characters outside the clause's DIGIT ZERO..DIGIT NINE repertoire. The round trip is not byte-preserving either — the encoder re-emits ASCII. |

### The C twins

| # | Where | Defect |
|---|---|---|
| A9 | `bcir_asn1_streampack.c` | `sp_find` matched a context tag's class and number but **not its form**, so an EXPLICIT encoding was accepted where the schema says IMPLICIT. Callers read `node.content` directly, so the nested TLV's own identifier and length octets became the projected `sourcePlan` string. |
| A10 | `bcir_per.c` | The semi-constrained reader clamped a negative lower bound to zero, discarding real headroom: the largest value of `INTEGER (-1..MAX)` — offset 2⁶³, sum exactly `INT64_MAX` — was refused `BCIR_PER_RANGE`. A **false rejection** of a conforming peer. |
| A11 | `bcir_asn1.c` | The X.690 length accumulator was a `size_t`. `BCIR_ASN1_MAX_LENGTH_OCTETS` is 8, so on the 32-bit targets these freestanding twins are written for, `0x0000000100000005` wrapped to `5` and the bound check passed against a length *smaller* than declared. |

A11 is invisible on a 64-bit host, which is why the suite never caught it. It is the one
finding here that is purely a portability defect and purely a security one.

---

## 3. Class B — vacuous checks

All fixed in `46b097e`.

| # | Law | What passed clean |
|---|---|---|
| B1 | R9 | **An atomic realized as a vector or a gather.** Candidate generation dispatched on `stride_class` alone, so `ATOMIC_ADD` was offered `U vec16` under SCALAR and `GGG gather` under RANDOM. Both `verify()` and `verify_plan()` returned clean. A vectorized read-modify-write is a data race, and the lost ordering and synchronization cost was unpriced. |
| B2 | R9 | **A fabricated candidate.** `Candidate(Lane.U, width=3, name="forged", cost=0)` for `vector_add`: a width the hardware cannot issue, a name denoting no realization, zero cost. The law checked lane against geometry and score against the step sum — and zero cost satisfies the score check trivially. |
| B3 | R10 | **An empty pack.** Every provenance loop iterated `pack.segments`, so a pack realizing none of the module's claims verified clean. |
| B4 | R10 | **A pack from a different plan.** `verify_all` checked plan and pack independently and never proved the pack was derived from that plan, so a pack hydrated from a `vec4` plan verified clean against a `scalar` one. The graph → plan → pack chain had no link. |
| B5 | `api.py` | **`attested=True` on an illegal module.** `build_artifact` set the field from R12 alone while it reads as "this artifact is legal", so a module violating R5 came back attested with an empty diagnostic tuple. A faithful lowering of an illegal plan is exactly what a deployable-artifact API must not bless. |
| B6 | R12 | **A miscompile with a clean attestation.** Both emitters write `C[i] = A[i] op B[i]` unconditionally; a claim declaring `offset=8` or `stride_k=4` was accepted, lowered to that same body, and verified clean. The kernel computes a different function than the claim declares. |

B1 and B2 were fixed on **both** sides — generator and law. Fixing only the generator would
leave R9 unable to call a tampered or third-party plan illegal, and R9 is what a certificate
rests on. B6 was fixed at the one shared selection gate, which is also why the defect appeared
identically in the LLVM and C rails.

---

## 4. The one finding left half-closed

**Provenance digest collisions (B7).** `hash_target` omits the memory hierarchy. Scaling the
DRAM tier's bandwidth and latency factors by 32 moves a `vector_add(4096)` plan's score from
**51,200 to 1,574,912 with the digest unchanged**. `hash_module` also sorts claims by id, so
two claims declared `a, b` and `b, a` — which plan to scores 3840 and 4352 — share a digest.

This is the most direct blocker to a content-addressed TMSAO scope, and it is **not fixed
here**, for a stated reason: `BCIRVerifyPass.cpp`'s `hashTargetFromIR` and `hashModuleFromIR`
recompute these hashes field for field from the IR for R13's cross-check, and
`TargetCapabilityOp` carries no ODS attribute for the tiers. Closing it means new ODS
attributes plus a matching C++ walk, landed together — and the MLIR rail does not build on
this host (LLVM 18 against a tree targeting 22). Landing a Python-only change would put the
two rails into silent disagreement about a content address, which is worse than the gap.

**What is closed is the consequence.** `replay()` compared only the digest, so it returned a
plan scoring 1,574,912 for a manifest recording 51,200 and raised nothing — while
`reproduces()` on the same inputs correctly returned `False`. `replay()` now compares the
produced plan as well, so a gap in the hash surfaces as a loud `ProvenanceMismatch` rather
than a wrong answer. A known-incomplete hash is survivable; a replay that silently answers
with a different plan is not.

This is the first item of the GEM+ scope work.

---

## 5. Not reproduced

Two items from the source report were investigated and are **not** carried forward as
confirmed:

- **CSE merging effectful claims.** The signature in `realize.py` is narrower than a full
  semantic identity, and the concern is sound in principle, but no reproduction was
  constructed here. Recorded as unverified rather than claimed.
- **The decoupled GGG tail overlapping barriered work.** Same: the scheduling code does build
  main-stream dependencies before splitting, and a concrete failing module was not produced
  within this pass.

Both remain open questions worth a targeted attempt, not findings.

---

## 6. Verification

```
quick tier      3299 passed, 0 failed      (3280 before; +19 new regression tests)
c-runtime tier  green
C twins         gcc -Wall -Wextra -Werror clean, all three
ruff            18 errors, unchanged from 3decf69 — none introduced
docs governance status / links / retired paths / claims all pass
```

The MLIR rail is unbuilt on this host and is CI's to verify; no ODS or pass source was
changed in either commit, so R13's cross-check constants are untouched.

New regression modules, both registered in `run_all.py`:

- `bcir/tests/test_asn1_canonical_bytes.py` — 11 tests, Class A
- `bcir/tests/test_verify_fail_closed.py` — 8 tests, Class B

Each drives the *failure*, not only the success. A test asserting the honest path would have
passed against all fifteen defects.

---

## 7. Recommended next

1. ~~**Sweep for the remaining host-parser instances.**~~ **Done** — see
   [`BCIR_SECURITY_AUDIT_2026-08-12b.md`](BCIR_SECURITY_AUDIT_2026-08-12b.md). It found five
   more, all in XER and JER, and refined the rule: the vulnerable predicate is every
   Unicode-aware host predicate (`str.isdigit()`, a regex `\d`), and it bites only on the
   TEXT rails, because the octet rails decode contents as ASCII first.
2. **Close the provenance hash across both rails** — ODS attributes for the memory hierarchy
   and the claim-order fields, plus the matching `hashTargetFromIR`/`hashModuleFromIR` walk.
   This is the P−1 correctness-closure item and it gates any content-addressed TMSAO claim.
3. **R11's generation tracking.** The pack records only the *maximum* `map_gen`/`data_gen`
   across resources, so bumping one resource below another's maximum leaves the pack "not
   stale" — reproduced, unfixed, because it needs a per-resource version vector or a digest
   in the StreamPack format, which changes the ASN.1 projection and its C twin.
4. **Attempt the two unreproduced findings** properly, and either confirm or retire them.
