# BCIR independent security sweep, 2026-08-12 (second pass)

A deliberately independent pass over `main` at `ac5a71c`, run *after* the findings from the
Codex report were closed. The brief was to find what that report did not cover, so nothing
here is inherited: each area was attacked on its own terms and each finding reproduced before
it was fixed.

**Result: 7 defects confirmed and fixed, 3 candidate findings investigated and cleared, 4
areas swept clean.** The clean results are reported at the same weight as the hits — an audit
that lists only what it found gives a reader no way to judge its coverage.

---

## 1. What the previous pass handed this one

The 2026-08-12 audit named a root cause — *a wire format parsed with a host-language parser* —
and recommended a mechanical sweep for the rest. That sweep is §2 below, and it was the
highest-yield thing in this pass: **five of the seven findings came from it.**

It also refined into something sharper than "don't call `int()`". The vulnerable predicate is
not only `int()` and `float()`; it is **every Unicode-aware host predicate** — `str.isdigit()`,
`str.isnumeric()`, and a regex `\d` or `\w` without `re.ASCII`. And there is a clean rule for
where it bites:

> **The octet rails are safe; the text rails are not.** X.690, X.691 and X.696 decode contents
> octets as ASCII before anything numeric happens, so a non-ASCII digit never reaches the
> parser. `decode_utctime` refuses `١٢٣٤٥٦٧٨٩٠١٢Z` at the ASCII decode and never consults its
> `^\d{12}Z$` regex at all — the regex *is* Unicode-aware, and it does not matter. XER and JER
> carry UTF-8 by design, so they have no equivalent earlier gate.

That rule is what made the sweep finite: it predicted the two files the remaining defects were
in, and both predictions held.

---

## 2. Confirmed and fixed

### 2.1 The text rails read Python's digits, not X.680's (5 defects)

X.680 §12.26 spells an arc out as *"an arbitrarily long sequence of ISO/IEC 10646 characters in
the range 0 (DIGIT ZERO) to 9 (DIGIT NINE)"*, and adds that it *"shall not commence with a 0
(DIGIT ZERO) character unless it has only a single character"*.

| # | Clause | Accepted |
|---|---|---|
| 1 | X.680 §19.9 | `<S>٤٢</S>` → the INTEGER 42. Also FULLWIDTH `４２`. |
| 2 | X.680 §12.9 | The same for a `realnumber`: `١.٥` → 1.5. |
| 3 | X.680 §9.8 | The same for an XER object identifier — **and** `1.2.0840`, a second spelling of `1.2.840`. |
| 4 | X.697 §32 | Both defects again in JER's object-identifier string. |
| 5 | X.680 §12.15.8 | The numeric character escape. **Nine** byte strings decoded to `A`. |

Finding 5 is the widest: `int(digits, 16 if x else 10)` accepts PEP 515 underscores, a leading
PLUS SIGN, surrounding whitespace, every Unicode decimal digit, and — because `int(s, 16)`
strips one — a **second `0x` prefix**. So `&#65;`, `&#x41;`, `&#6_5;`, `&#x4_1;`, `&#+65;`,
`&# 65 ;`, `&#٦٥;`, `&#６５;` and `&#x0x41;` were all the character `A`.

**The tell that these were not near-misses.** `_parse_integer` refuses a leading zero and cites
X.680 §12.8 for it; `_parse_oid`, three functions later in the same file, does not — though the
arc production carries the same rule. Five sites each reached for a host predicate
independently. So the fix is **one predicate they now share** (`is_ascii_digits` /
`is_number_form` in `values.py`), not five local repairs — five local repairs is how there came
to be five.

JER's own INTEGER path was already correct: its scanner enforces the JSON number grammar
*before* converting. That is the pattern the rest now follows, and it was already in the tree.

### 2.2 A certificate named the run that produced it (2 defects)

Both in `select_certified`, and both the same mistake: `repr` is a debugging rendering, and it
was being used as a content address.

**`schema_digest` depended on a heap address.** `Component.default`'s sentinel was a bare
`object()`, so every component declaring no DEFAULT reprs as `<object object at 0x7f...>` —
which means **every SEQUENCE and every CHOICE** carried an address into `repr(kind)`. The
sentinel is a module-level singleton, so the digest was stable *within* a process and different
on every run:

```
run 1: 8d5901288308c058aff4ab726025621f
run 2: 6aef492f94dc239959116c4095ac2b82
run 3: ecc33be79c72ae1b3a88d94d360b3fb3
```

Same program, same schema, three certificates. The in-process stability is precisely why no
test caught it, so the regression test spawns subprocesses.

**`value_digest` depended on dict insertion order.** A SEQUENCE value is a mapping from
component name to value, and a Python dict's `repr` follows insertion order. `{"a": 1, "b": 2}`
and `{"b": 2, "a": 1}` — equal as values, encoding to the byte-identical `3006800101810102` —
produced two different digests, and therefore two certificates for one value.

The fix is not to sort the dict. It is to digest the **canonical octets**, which is what BCIR
digests everywhere else and is canonical by construction. A value nothing can encode still gets
a (no-winner) certificate, so that case is domain-separated rather than falling back to `repr`.

For a system whose premise is certified, content-addressed, replayable artifacts, these two are
the most severe findings in either pass: the address one means **no certificate was reproducible
at all**.

---

## 3. Investigated and cleared

Reported at equal weight, because "we looked and it was fine" is information.

**Three `-fsanitize=integer` hits in the C twins are not defects.** Fuzzing under clang with
`-fsanitize=integer` — a check the repo's gcc-based gate *cannot run* — flagged
`bcir_asn1.c:441`, `bcir_per.c:107` and `bcir_oer.c:117`. All three are deliberate, commented,
and **well-defined**: C99 §6.5.7p4 makes unsigned left shift modular by definition and §6.2.5p9
does the same for unsigned subtraction, and `bcir_per.c`'s comment states the wrap is the
technique ("*so that lb = INT64_MIN and ub = INT64_MAX does not overflow the subtraction*").
Replaying each crash input under ASan+UBSan gives **zero** hits while the integer build gives
one, which is the clean discriminator.

Worth recording as a *harness* note rather than a code note: enabling `-fsanitize=integer` in CI
would need suppressions for these three, and without them the signal would be noise.

**The JER frame is not a decompression surface.** `jer_bounded` imports `zlib`, which looked
like a bomb risk; it uses only `crc32`, and `unframe` requires an exact length match before
returning any payload.

**Signature comparison is timing-safe.** The staged loader uses `hmac.compare_digest`.

---

## 4. Swept clean

| Area | Method | Result |
|---|---|---|
| X.690 primitive decoders | 24 hand-built edge encodings: OID non-minimal subidentifiers and unbounded arcs, RELATIVE-OID, BIT STRING unused-bit counts > 7 and on empty content, INTEGER §8.3.2 minimality, BOOLEAN/NULL content length | every illegal form correctly refused |
| XER entity handling | billion-laughs / general-entity probes | no general entity mechanism exists; only `amp`/`lt`/`gt` and the numeric escape, and DTD, CDATA and processing instructions are refused by name |
| ASN.1 C decoders | ~30M executions under ASan+UBSan (`bcir_asn1`, `bcir_per`, `bcir_oer`, `bcir_xer`) | **0 sanitizer hits** |
| Other `object()` sentinels | reachability from any digest | `_PENDING_COMPONENT`, `_UNREPRESENTABLE` and the `lower.py` marker reach no compiled schema |

The X.690 primitive layer is the strongest thing either audit examined — it refused every edge
case put to it on the first attempt.

---

## 5. Verification

```
quick tier   3311 passed, 0 failed   (3299 before; +12 new regression tests)
fuzzing      ~30M execs, 4 decoders, ASan+UBSan, 0 hits
ruff         clean on every changed file
```

New regression modules, both registered in `run_all.py`:

- `bcir/tests/test_asn1_text_rails.py` — 6 tests
- `bcir/tests/test_certificate_identity.py` — 6 tests

No C, ODS or MLIR source changed in this pass.

---

## 6. What remains open

Unchanged from the first pass, and both are GEM+ scope work rather than patches:

1. **The provenance hash omits the memory hierarchy** — a two-rail change (ODS attributes plus
   `hashTargetFromIR`).
2. **R11 tracks only the maximum resource generation** — needs a version vector or digest in
   the StreamPack format.

Newly noted here:

3. **CI cannot see the integer-sanitizer class.** Adding it needs the three suppressions above.
   Low priority — that class is well-defined behaviour — but it is a real blind spot, and it is
   the same blind spot that hid the 32-bit length wrap fixed in the first pass.
