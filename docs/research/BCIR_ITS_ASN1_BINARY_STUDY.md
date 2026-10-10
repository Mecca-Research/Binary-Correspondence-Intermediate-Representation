# The ITS security envelope: ASN.1 against hand-made binary, re-measured (2026-10-09)

> Dated, non-normative. A re-examination of Bittl, Gonzalez, Spähn and Heidrich, "Performance
> Comparison of Data Serialization Schemes for ETSI ITS Car-to-X Communication Systems",
> *Int. J. Advances in Telecommunications* 8(1&2), 2015, with BCIR's ASN.1 machinery. Everything
> below is reproducible from the repository (§10); the code is `bcir/asn1/its_security.py`,
> `its_study.py`, `its_native.py` and `cgen.py`, and the witnesses are `test_its_security.py`
> and `test_asn1_cgen.py`. Counts live only in the generated [`STATUS.md`](../STATUS.md).

## 0. The answer

The 2015 study concluded that "binary encoding greatly outperforms ASN.1 encoding in the clear
majority of cases for the security envelope", that binary was "significantly faster", and that
the security envelope's standardization "should reconsider the recent shift from binary encoding
towards usage of ASN.1". On the same content, measured here:

* **Size: ASN.1 is smaller than binary in every case.** The V1.1.1 data model under canonical
  UNALIGNED PER, transcribed one construct at a time, is 92/212/223/220 octets against binary's
  96/222/233/230. With V1.1.1's own prose rules moved into the schema (the TMSAO schema, §2.3)
  it is 86/203/213/212: 7.8–10.4% below binary in all four profiles. On the paper's own Table VII
  metric (the CAM stream average) that is **97.7 octets against binary's 108.6 and the paper's
  winner, optimized EXI, at 98.4**.
* **Time: an ASN.1 codec is faster than binary in every case.** Schema-compiled C for
  CANONICAL-OER on the TMSAO schema encodes and decodes every envelope in 0.55–0.74 of the time
  of a binary codec from the same generator, under clang 18, GCC 13 and clang 23 alike, and
  executes 0.60–0.73 of its instructions. That same encoding is *also* smaller than binary in all
  four profiles (93/216/226/224; average 105.3).
* **Memory: no heap at all.** The generated codecs reference no allocator (checked with `nm`);
  decoded variable data lives in a caller's arena (0 octets for the CAM envelope under
  TMSAO-COER), and the deepest stack measured is 1,008 octets, against the paper's 12–20 KB of
  stack and 0.2–4.3 KB of heap.

So there is no gap left to close on the paper's metrics, and the overturn does not rest on one
point: TMSAO-COER is a single ASN.1 encoding that is smaller, faster and lighter than binary on
every envelope. What remains is a trade inside ASN.1. The smallest encoding (UPER on the PER
instance) costs 1.1–1.8x binary's time, because its 32-octet keys, digests and signatures sit at
arbitrary bit offsets (§4.3).

## 1. The 2015 study

The paper compared, for CAM, DENM and the TS 103 097 security envelope, a binary encoding, ASN.1
UPER, Protocol Buffers and EXI, on an AMD Geode LX (500 MHz), an Intel Atom Z520PT and an Intel
Core i7-2640M, everything built with GCC 4.8.2 at `-O3`. The ASN.1 codec was FFASN1's, the binary
one ezCar2X's.

The security envelope's two encodings did not encode the same thing:

| column | data model | rule | codec |
|---|---|---|---|
| binary | TS 103 097 **V1.1.1** (TLS-style presentation language) | V1.1.1's own byte rules | ezCar2X, hand-written |
| ASN.1 | a **draft** of the ASN.1 successor (the paper's reference [15]) | UPER | FFASN1, table-driven |

Table VI (encoded length, octets) and Table VII (the average CAM envelope at 10 Hz with the
certificate at 1 Hz, `(9 s_w/o + s_w) / 10`):

| | p1 (no cert.) | p1 (cert.) | p2 | p3 | Table VII |
|---|---|---|---|---|---|
| binary | 96 | 222 | 233 | 230 | 108.6 |
| ASN.1 UPER (draft) | 88 | 240 | 249 | 247 | 103.2 |
| protobuf | 133 | 306 | 318 | 312 | 150.3 |
| EXI | 90 | 210 | 215 | 213 | 102.0 |
| EXI (optimized schema) | 87 | 201 | 206 | 204 | 98.4 |

The paper's own text explains one difference: profiles 2 and 3 have the same ASN.1 length
because the draft makes `messageType` mandatory where V1.1.1 makes it optional. The protobuf
and EXI schemas were derived from the draft ASN.1 definitions (the paper's §V-B). So three
things changed at once between the columns: the data model, the encoding rule and the codec.
This study holds the data model fixed and varies the other two.

## 2. Method: one abstract value, every encoding

### 2.1 V1.1.1's binary rules, reconstructed and held to the paper

TS 103 097 V1.1.1's text could not be fetched: `www.etsi.org` is blocked by this environment's
egress policy (the proxy answers 403). The binary rules are therefore a **reconstruction**,
held to the paper's published numbers rather than trusted. `its_security.encode_binary` must
reproduce all four Table VI binary lengths exactly, and it does (`its.reconstruction.mismatch`
0). The free choices, each pinned by a number:

* five top-level fields (version, profile, header list, payload, trailer list): the paper's
  Table III gives 5 at nesting level one, and only this arrangement makes the CAM envelope 96;
* `message_type` a uint16 header field: profiles 2 and 3 differ only in it, by 3 octets;
* `generation_location` a 10-octet ThreeDLocation: DENM is 233 against 222;
* a `<var>` vector's length an IntX (n leading one bits, a zero, then 7 + 7n value bits), so
  the certificate envelopes' header list takes a two-octet length. The certificate is then 133
  octets, which is exactly a minimal V1.1.1 authorization ticket: a compressed-y verification
  key (the paper's Listing 1), an assurance level, one ITS-AID with a 3-octet SSP, start and end
  times, and an x-only ECDSA signature.

Everything else is the presentation language's plain reading: a uint8 type code before every
`select`, big-endian uintN, and fixed opaques raw. The payload is the one-octet dummy the paper
used, with mandatory fields only and real-looking times and coordinates.

### 2.2 The transcription

`ITS-SecuredMessage-V111.asn1` is the V1.1.1 data model in ASN.1, one construct per construct. A
field list is a `SEQUENCE OF` a `CHOICE`, an `opaque<var>` an `OCTET STRING`, uintN an
`INTEGER (0..2^N-1)`, and IntX `INTEGER (0..2^56-1)`. One Python value of this module is encoded
by the byte rules above and by BCIR's UPER, APER, COER and DER rails. A difference of one octet
between two columns is therefore a difference of one octet of encoding.

### 2.3 The TMSAO schema

`ITS-SecuredMessage-TMSAO.asn1` is the Theoretical-Maximum pass on the schema. It carries every
V1.1.1 value, with what V1.1.1's prose states and its presentation language cannot moved into the
type:

* **Field lists become sets.** The prose allows each header field, trailer field, subject
  attribute and validity restriction at most once, in ascending type order (signer information
  first). A list with those rules is a set of optional fields in a fixed order, which a
  `SEQUENCE` of `OPTIONAL` components carries exactly: one presence bit replaces a uint8 type
  code, and the list's length disappears. `to_tmsao`/`from_tmsao` are the bijection between
  V1.1.1's values and their images, and they *refuse* a list the prose forbids rather than
  reorder it, since reordering would map two binary encodings to one value.
* **Nothing is narrowed.** Every opaque keeps its size, every CHOICE its alternatives and its
  extension marker, and every INTEGER its whole V1.1.1 range. The converse does not hold in the
  schema: a shaped integer (below) is extensible, and BCIR reads an extensible constraint by its
  root, so the module and the codecs generated from it admit integers past V1.1.1's range. The
  mapping holds the range instead -- both directions refuse a value the V1.1.1 binary rules
  cannot carry (`test_the_tmsao_mapping_refuses_a_value_v111_cannot_carry`).
* **Some integers are shaped**: an extensible constraint whose root is the range the protocol
  produces, with the rest of V1.1.1's range as an extension addition. IntX is shaped everywhere
  (root 0..127). The rest depend on the rule, through X.683. `SecuredMessage` keeps V1.1.1's
  fixed widths (OER's best), and `SecuredMessagePer` shapes the version and profile
  (root 0..7), the generation time (root 0..2^53-1 µs, every instant before the year 2289) and
  the message type (root 0..255).

### 2.4 The accounting

`its_study.framing` partitions every bit of an envelope into **content** (the fields' own
values) and four kinds of **framing**: selectors (type codes, CHOICE indices), lengths and
counts, presence and extension bits, and padding. It encodes each subtree with the rule's own
encoder and attributes the difference between a node and its children to the node's kind. The
parts sum to the encoding's length exactly, for every codec and profile
(`test_the_bit_partition_is_exact_for_every_codec_and_profile`).

### 2.5 The codecs

`bcir.asn1.cgen` compiles an ASN.1 module to straight-line C, one function per type per rule,
for COER, UPER and byte-oriented rules (`ByteRule`: per-CHOICE type codes and the IntX bound).
The V1.1.1 binary codec is therefore *generated from the transcription by the same compiler*
as the ASN.1 codecs, with the same value model and the same I/O primitives. A time difference
between two columns is a difference of encoding rule, not of how much care one codec got.

Shared by all of them:

* the value model: an INTEGER is its narrowest C integer, a fixed OCTET STRING an inline array,
  and a variable one a pointer and a length (zero-copy where the rule is octet-aligned, copied
  into the arena under UPER);
* decoded variable data goes in a caller-supplied arena, never the heap;
* a depth bound refuses cyclic values;
* the decoders are canonical-only: an accepted input re-encodes to itself.

### 2.6 Measurement

`its_native` builds one harness per compiler, with `-O2`, over both generated files. Timing
takes the median of rounds of thousands of iterations. The final tables are the median of fifteen
interleaved repetitions per compiler, each of 24 rounds × 3,000 iterations. **Code placement moves
these numbers more than repetition does.** Two builds whose codecs differ only in an unrelated
function timed byte-identical codecs up to 17% apart, interleaved in one window, while their
instruction counts did not move at all. So every timing ratio below carries a band of that width,
and the instruction ratios (§4.2), which carry none, are the claim's firm footing. Every case is
encoded, decoded and re-encoded to its own octets once before any clock starts, so a refusal is
never timed as a decode. A `memcpy` of the encoding is timed beside each
codec as the memory roofline. Because this host is a shared virtual machine with no PMU,
instruction counts from callgrind (deterministic per compiler) complement the clock. Memory has
three measures:

* heap, by an `nm` scan for allocator symbols;
* arena octets, exactly;
* stack, by a painted high-water mark on a thread whose stack the harness owns (192 octets is
  the measurement's floor).

## 3. Size

### 3.1 Every encoding, every profile

| encoding | p1 | p1cert | p2 | p3 | Table VII |
|---|---|---|---|---|---|
| **V1.1.1 binary** (reconstruction = paper) | 96 | 222 | 233 | 230 | 108.6 |
| transcription, UPER | 92 | 212 | 223 | 220 | 104.0 |
| transcription, APER | 93 | 218 | 230 | 227 | 105.5 |
| transcription, COER | 98 | 227 | 238 | 235 | 110.9 |
| transcription, DER | 115 | 279 | 297 | 294 | 131.4 |
| TMSAO, UPER | 89 | 206 | 216 | 214 | 100.7 |
| TMSAO, APER | 90 | 210 | 222 | 220 | 102.0 |
| TMSAO, COER | 93 | 216 | 226 | 224 | 105.3 |
| TMSAO PER instance, UPER | **86** | **203** | **213** | **212** | **97.7** |
| TMSAO PER instance, COER | 95 | 218 | 228 | 226 | 107.3 |
| paper: ASN.1 UPER (draft model) | 88 | 240 | 249 | 247 | 103.2 |
| paper: EXI (optimized, draft model) | 87 | 201 | 206 | 204 | 98.4 |

* The **transcription alone** beats binary under UPER in all four profiles, where the paper's
  draft-schema UPER lost three of four. The data model, not ASN.1, made the paper's ASN.1
  column long.
* Every TMSAO encoding but DER is shorter than binary in every profile, COER included. DER is
  BCIR's digest form, not a transport choice.
* Against optimized EXI: the PER instance's UPER is smaller on the CAM envelope (86 against 87)
  and on Table VII's average (97.7 against 98.4), and 2–8 octets larger on the certificate,
  DENM and generic envelopes. EXI's columns encode the draft's data model, so that comparison is
  not same-content. The binary comparisons are.

### 3.2 Where the octets go

The bit partition for the CAM envelope (p1) and its certificate variant:

| p1 | total bits | content | selector | length | presence | padding | framing |
|---|---|---|---|---|---|---|---|
| V1.1.1 binary | 768 | 680 | 64 | 24 | 0 | 0 | 11.5% |
| transcription, COER | 784 | 680 | 64 | 40 | 0 | 0 | 13.3% |
| transcription, UPER | 736 | 680 | 28 | 24 | 0 | 4 | 7.6% |
| TMSAO, COER | 744 | 680 | 32 | 8 | 24 | 0 | 8.6% |
| TMSAO, UPER | 712 | 680 | 12 | 8 | 12 | 0 | 4.5% |
| TMSAO PER instance, UPER | 688 | 651 | 12 | 8 | 16 | 1 | 5.4% |

| p1cert | total bits | content | selector | length | presence | padding | framing |
|---|---|---|---|---|---|---|---|
| V1.1.1 binary | 1776 | 1567 | 136 | 73 | 0 | 0 | 11.8% |
| transcription, COER | 1816 | 1568 | 136 | 112 | 0 | 0 | 13.7% |
| transcription, UPER | 1696 | 1563 | 56 | 72 | 1 | 4 | 7.8% |
| TMSAO, COER | 1728 | 1568 | 72 | 48 | 40 | 0 | 9.3% |
| TMSAO, UPER | 1648 | 1562 | 25 | 32 | 26 | 3 | 5.2% |
| TMSAO PER instance, UPER | 1624 | 1533 | 25 | 32 | 30 | 4 | 5.6% |

The content is the same octets in every column: keys, digests, signatures and times. The
columns differ in framing. V1.1.1 frames in whole octets: eight one-octet type codes and three
one-octet lengths on the CAM envelope, 88 bits around 680. The TMSAO schema under UPER frames
the same 680 bits with 32: a presence bitmap with its extension bits, CHOICE indices and one
length. The PER instance then carries the version, profile, time and message type in 29 fewer
content bits, for four more extension bits. Its content goes *below* V1.1.1's field widths.

**Against the content floor.** With no framing at all, the fields at V1.1.1's own widths are
85/196/206/204 octets (`its_study.content_floor`), averaging 96.1 on Table VII. The PER instance's
UPER is 1/7/7/8 octets above that floor, and 97.7 against 96.1 on the average
(`its.size.uper.avg`). Shaped integers carry some fields in fewer bits than V1.1.1's widths, so
this is a measured floor, not a lower bound on every encoding (TMSAO-3, not TMSAO-1). The framing
that remains (37 bits on the CAM envelope) is the extensibility and optionality V1.1.1's own data
model carries.

### 3.3 BCIR's optimizer reaches the same answer

ASN.1 Phase H's cost-governed selector (`bcir.asn1.selection`) decides legality first, keeps
canonical rules only, and then optimizes the objective. Given the PER instance's values and the
paper's objective, octets on the channel, it selects canonical UNALIGNED PER on every profile, at
exactly the sizes above. With no objective it keeps DER, the pinned degenerate case. Every rule
BCIR speaks, JER and BER included, carries all four values
(`test_kbcir_selects_the_studys_smallest_encoding_for_the_channel`). The selector's latency
objectives read Python-oracle timings and are indicative; the C timings of §4 are the evidence
for time.

## 4. Time

### 4.1 Nanoseconds per envelope, encode / decode

Medians, `-O2`, this container's x86-64 vCPU.

**clang 18.1.3**

| codec | p1 | p1cert | p2 | p3 |
|---|---|---|---|---|
| **V1.1.1 binary** | 30.7 / 23.5 | 74.4 / 63.9 | 77.7 / 66.8 | 75.7 / 65.7 |
| transcription, COER | 22.4 / 20.8 | 55.5 / 53.7 | 58.5 / 55.9 | 56.4 / 55.0 |
| transcription, UPER | 33.9 / 43.8 | 84.6 / 115.8 | 91.7 / 121.5 | 90.2 / 118.4 |
| **TMSAO, COER** | **18.2 / 16.6** | **44.3 / 39.7** | **44.9 / 40.0** | **45.8 / 38.6** |
| TMSAO PER instance, UPER | 35.8 / 39.9 | 85.6 / 95.1 | 92.4 / 100.4 | 92.2 / 102.2 |

**GCC 13.3**

| codec | p1 | p1cert | p2 | p3 |
|---|---|---|---|---|
| **V1.1.1 binary** | 27.1 / 22.8 | 74.9 / 55.0 | 78.1 / 56.1 | 76.9 / 54.6 |
| transcription, COER | 25.2 / 23.4 | 57.0 / 54.6 | 62.3 / 56.5 | 60.8 / 55.6 |
| transcription, UPER | 39.8 / 38.8 | 94.5 / 101.8 | 104.9 / 105.5 | 101.7 / 104.9 |
| **TMSAO, COER** | **19.8 / 14.2** | **41.3 / 34.9** | **43.6 / 38.2** | **45.0 / 37.1** |
| TMSAO PER instance, UPER | 34.6 / 38.5 | 79.6 / 91.8 | 85.9 / 96.7 | 87.5 / 94.5 |

**clang 23.1.3**

| codec | p1 | p1cert | p2 | p3 |
|---|---|---|---|---|
| **V1.1.1 binary** | 24.7 / 22.2 | 67.7 / 61.7 | 70.8 / 64.6 | 69.1 / 62.4 |
| transcription, COER | 22.1 / 20.0 | 53.8 / 49.7 | 56.1 / 52.5 | 54.5 / 51.3 |
| transcription, UPER | 34.8 / 42.5 | 85.8 / 106.1 | 92.7 / 115.1 | 90.6 / 110.9 |
| **TMSAO, COER** | **18.4 / 16.2** | **42.6 / 39.3** | **43.4 / 39.7** | **43.1 / 39.6** |
| TMSAO PER instance, UPER | 34.4 / 40.0 | 84.2 / 95.4 | 92.2 / 101.8 | 91.0 / 100.1 |

TMSAO-COER over binary, per operation: 0.58–0.71 (clang 18), 0.55–0.73 (GCC 13), 0.61–0.74
(clang 23). Even the *transcription* under COER, the same model with the type codes turned into
CHOICE tags, takes 0.73–1.02 of binary's time: less in every case under both clangs, and under
GCC less on every encode and within 3% of parity on the decodes (0.99–1.03).

The absolute figures are not comparable with the paper's: different hardware, compiler and
decade, and the paper's figures are on log axes in microseconds. The ratios are comparable,
because both sides of each ratio were built by one generator and timed in one run.

### 4.2 Instructions per envelope (callgrind)

| codec | GCC 13: p1 | GCC 13: p1cert | clang 18: p1 | clang 18: p1cert |
|---|---|---|---|---|
| V1.1.1 binary | 454 / 435 | 1196 / 1086 | 413 / 405 | 1073 / 1074 |
| transcription, COER | 351 / 398 | 836 / 969 | 354 / 394 | 909 / 1034 |
| transcription, UPER | 650 / 707 | 1534 / 1755 | 591 / 768 | 1506 / 1894 |
| **TMSAO, COER** | **314 / 282** | **720 / 719** | **301 / 272** | **738 / 760** |
| TMSAO PER instance, UPER | 620 / 655 | 1433 / 1541 | 616 / 674 | 1502 / 1588 |

**Why COER wins.** A presence bitmap replaces a type-code dispatch per field. Every 32-octet key
and signature is octet-aligned, so it moves as a `memcpy`. Fixed-width integers are single
byte-swapped loads. The binary rule pays a type code and a branch per field, and an IntX parse
(a leading-ones count) per list.

### 4.3 What the smallest encoding costs

UPER on the PER instance runs at 1.06–1.80x binary's time (1.20–1.66x its instructions). Its
content is mostly 32-octet keys, digests and signatures at arbitrary bit offsets, and each eight
octets of such a run costs a load, two shifts and a store, where an aligned rule copies. The
line profile puts most of the remaining decoder time in exactly that loop and in the per-field
bit reads. ALIGNED PER is the measured middle point: on the TMSAO schema it is 90/210/222/220
octets (average 102.0, smaller than binary everywhere) and aligns every such run to an octet.
`cgen` has no APER backend yet (§11).

### 4.4 Against the roofline

TMSAO-COER on the certificate envelope encodes in 4.5x and decodes in 4.5x the time of one
`memcpy` of its 216 octets (`its.coer.encode.floor`, `its.coer.decode.floor`). Its encode +
decode sits at 0.64 of binary's, with the two-copy roofline at 0.14 (`its.coer.vs.binary`). UPER
sits at 1.41 against the same kind of roofline (`its.uper.vs.binary`). Those are the rows' frozen
values; the final build measures 0.60 and 1.31. Both rows divide one compiled codec by another,
so they are host-dependent: graded on the baseline host, INDICATIVE elsewhere.

## 5. Memory

| codec (clang 18) | heap | arena p1 / p1cert | stack enc/dec p1cert |
|---|---|---|---|
| V1.1.1 binary | 0 | 1,200 / 1,928 | 464 / 376 |
| transcription, COER | 0 | 684 / 1,180 | 328 / 312 |
| transcription, UPER | 0 | 688 / 1,192 | 920 / 656 |
| **TMSAO, COER** | 0 | **0 / 512** | **248 / 232** |
| TMSAO PER instance, UPER | 0 | 1 / 516 | 984 / 544 |

Under GCC 13 the stacks are smaller still: at most 479 octets, and TMSAO-COER 223/247. The
transcription's arena holds its field lists (arrays of CHOICE structs). The TMSAO schema has no
lists, so the CAM envelope decodes into the caller's struct alone, and the certificate case
allocates only the certificate (it sits behind a pointer because `SignerInfo` is recursive).

**V1.1.1's arena column is what its decoder reserves, not what its structs need.** The binary
codec fills the same C structs as the transcription's COER codec, so the structs need exactly
what COER takes: 684 / 1,180 octets. V1.1.1's vectors carry their length in octets with no
count, so a decoder cannot size a list's array before decoding the list. This one starts each
array at four slots and grows it in place while it is still the arena's last allocation, and the
capacity a short list leaves unused stays reserved. Both ways to size the arrays exactly cost
time: counting the elements first measured 1.5–1.8x the decoder's time, and giving the unused
slots back cost 3–9% more of its instructions. So the binary codec keeps the reservation: §4
times its fastest decoder, and this column shows what that decoder reserves.

The paper measured 240–4,327 octets of heap and 12,168–20,528 octets of stack per operation
for its binary and ASN.1 codecs.
Those are properties of the implementations it measured, not of the encodings. A
schema-compiled codec needs no heap at all.

## 6. The C refactor (C-PERF)

The user's request, refactor the C ASN.1 implementation for performance, covered two rails.

**The plan-driven PER decoder** (`runtime/c/bcir_per.c`, `bcir_per_plan.c`) reads its bit
fields an octet at a time and skips a string body rather than reading it bit by bit. A 200-octet
record went from 1.9 to 0.36 µs ALIGNED and from 2.0 to 0.27 µs UNALIGNED.

**The generated codecs** (`cgen`):

* Word-at-a-time bit I/O: the writer keeps its pending bits MSB-aligned in a 64-bit accumulator
  and flushes with one big-endian 8-octet store. The reader takes a 64-bit window per field, and
  an unaligned octet run moves eight octets per step.
* Each function's cursor lives in a local, and only the fields a call can change (`p`, `acc`,
  `n`) are synchronized at call boundaries.
* Every fixed-width integer is a single byte-swapped load or store.
* **The per-field primitives are forced inline.** GCC's inliner reported `--param
  inline-unit-growth limit reached` on the generated files and left the bit writer out of line,
  so every field went through memory. Under GCC 13 that made UPER 1.8–3.8x slower than under
  clang.
  `always_inline` (behind a `__GNUC__` guard) fixed it.
* One bounds test serves the bit reader's fast path and its truncation check (eight readable
  octets hold at least 57 bits).
* The slow-path window takes values, not the reader. Taking its address there had made clang
  keep the decoder's cursor in memory: the same instruction count, 32–38% slower to decode.

Before and after, interleaved runs of the same harness, outputs identical:

| codec | GCC 13 encode / decode | clang 18 encode / decode |
|---|---|---|
| TMSAO PER instance, UPER | 0.30–0.34x / 0.56–0.62x | 0.64–0.68x / 0.57–0.60x |
| transcription, UPER | 0.24–0.26x / 0.36–0.40x | 0.76–0.81x / 0.60–0.65x |
| transcription, COER | 0.84–1.07x / 0.60–0.62x | 0.88–0.93x / 0.94–1.04x |
| TMSAO, COER | 0.92–0.99x / 0.72–0.85x | 0.91–1.02x / 0.89–0.96x |
| V1.1.1 binary | 0.97–1.05x / 0.54–0.86x | 0.88–0.94x / 0.86–0.95x |

The binary codec gained too: it is generated, so it gets every improvement the ASN.1 codecs
get, and the comparison in §4 stays like-for-like.

## 7. What the study found wrong in BCIR, and fixed

Adding the envelope to the corpus surfaced defects no existing test had reached. Each fix has a
witness that fails on the defect:

* **OER extensibility (OER-EXT), on four rails.** The Python OER rail, the E1 emitter, its C twin
  and the plan-driven C decoder wrote no X.696 §16.2.2 extension bit in an extensible SEQUENCE's
  preamble, and spelled §16.4/§16.5 extension additions and §20.2 extension alternatives as root
  ones. All four agreed with each other, which is exactly what a round trip cannot catch. The
  vectors are now cross-checked against an independent OER codec, asn1tools 0.169.0. Where the
  two disagree (asn1tools' OER flattens a version bracket into separate additions, which its own
  PER codec does not), BCIR follows X.691 §19.9's structure, and the test records why.
* **The native bench's field builders** read attributes that plan version 3 had removed, and
  timed decodes that failed. Every case is now decoded and checked once before it is timed.
* **The C PER plan decoder** gave a variable-size string's SIZE lower bound as 0 and aligned
  §17.8 lengths wrongly.
* **`cgen`'s helper selection** could keep a byte-order `#if` and drop its `#endif`, and dropped
  the cursor typedef from a file of fixed octets alone. Neither file compiled. Its macros also
  leaked unprefixed (`P_BSWAP16`, `P_NATIVE_WORDS`) into every generated source.
* **The stack harness** read a freed pointer (GCC 13's `-Werror=use-after-free`), and its bench
  timed decodes without checking that they succeeded.

**A review before merge** found ten more, in code no envelope reaches; each witness compiles a
small schema of its own:

* `cgen`'s decoders skipped the INTEGER value constraint its encoders check (variable-size forms
  only: a decoded value could fail its own re-encode). Its UPER preamble went out in one 57-bit
  `P_put` however many OPTIONALs there were. An extensible INTEGER with an open root bound did
  not compile. A SEQUENCE OF elements that encode in no bits under UPER (a one-enumerator
  ENUMERATED, a single-value INTEGER) refused its own encodings as truncated. The byte rule
  ignored a variable string's SIZE. Its vectors moved at every doubling, abandoning each copy.
  The new witnesses found one more: a function that never names its cursor did not compile
  under `-Werror`. A last re-read found another: the arena rounded its fill offset up to an
  element's alignment rather than the address it hands out, so caller memory starting off an
  8-octet boundary got misaligned arrays. It aligns the address now.
* The Python OER rail accepted, under CANONICAL, a set padding bit in the root preamble (the C
  plan decoder reports it as non-canonical), set unused bits in the additions' bitmap, and an
  addition sent equal to its DEFAULT: three second spellings.
* The V1.1.1 binary decoder copied the data's prefix for each vector element and read an
  element of no octets forever; it ignored SIZE as the byte rule did.
* The harness read `0g` as an octet. GEM+'s ITS timing ratios, which divide one compiled codec
  by another, were graded on every host; like the `native.*` rows they are host-dependent now.

**The witnesses can fail.** `tools/testing/faults/its.json` injects 29 of these defects, or their
inverses, into the oracle, both C twins, the generator and the harness, and each one is caught by
its own test (`tools/testing/red_sweep.py`). The sweep also found a gap. A bit reader that
over-reads its input and skips the truncation check is still refused by the decoder's final
length check, but as MALFORMED, so the damaged-input differential's accept/refuse verdict cannot
see it. The new witness holds every cut of every encoding to TRUNC.

## 8. ECN: binary is an encoding of the same abstract value

X.692 (ECN) is the part of the ASN.1 family for exactly this case: a hand-made bit layout stated
as an encoding of an ASN.1 type. V1.1.1's rules map onto it almost directly:

* type codes are an alternative determinant;
* fixed opaques are fixed octet spaces;
* IntX is a self-delimiting integer code. §19.7's `TO BITS` mapping exists for this, and its
  NOTE says it "is intended to support self-delimiting encodings of integers, such as Huffman
  encodings". IntX is eight contiguous code ranges, each of equal-length codes.

**BCIR's ECN rail cannot express IntX today.** It writes fixed encoding spaces and refuses
§21.2.7's `self-delimiting-values` (`ecn_syntax._EncodingSpace.width`), so an exact ECN replica
of V1.1.1 was not built. The byte rule is a `cgen.ByteRule` over the same ASN.1 module instead.
That reaches the study's point: the binary format is generated from the ASN.1 schema, and
"binary against ASN.1" is a choice of encoding rule for one ASN.1 type. Closing the ECN refusal
would make the same point inside X.692 (§11).

## 9. Threats to validity

* **The reconstruction is not the standard's text.** It reproduces all five published lengths,
  but two encodings of different content could share a length. The pins in §2.1 are the
  evidence that it does not, and a reader with access to ETSI's text can check them.
* **The binary codec is generated, not ezCar2X's.** That isolates the encoding rule. A
  hand-tuned binary codec could be faster than the generated one, and so could a hand-tuned
  COER codec.
* **Hardware.** One x86-64 vCPU on a shared virtual machine, with no PMU. The tables are
  medians of interleaved repetitions, and the instruction counts are deterministic. An embedded
  core (the paper's Geode) is not measured.
* **Code placement.** Relinking moved byte-identical codecs by up to 17% (§2.6). The weakest
  timing margin, TMSAO-COER at 0.74 of binary (the CAM encode under clang 23), survives that
  band, and the instruction ratio (0.60–0.73) does not depend on it.
* **Mandatory fields and a one-octet payload, as in the paper.** Optional header fields and
  real payloads would add the same content octets to every column.
* **The TMSAO schema is a schema for V1.1.1's content**, not today's TS 103 097 (whose later
  versions build on IEEE 1609.2). It answers whether ASN.1 can carry V1.1.1's content more
  compactly and faster than V1.1.1's binary, and it can.
* **Shaped roots are protocol knowledge.** A value outside a root still encodes, as an
  extension, so nothing is lost, but it costs more bits.
* The protobuf and EXI columns are the paper's, over the draft model. They are not re-measured.

## 10. Reproduction

```bash
python3 -c "from bcir.asn1 import its_study as st; print(st.sizes()); print(st.average_sizes())"
python3 -c "from bcir.asn1 import its_study as st; print(st.framing('tmsao-per-uper', 'p1'))"
python3 -m bcir.asn1.its_native /tmp/its gcc     # build the five codecs, print time and memory
python3 tools/perf/gemplus_baseline.py --compare --group its
python3 -m bcir.tests.run_all --tier quick -j 2  # test_its_security and test_asn1_cgen among them
```

`its_study.sizes` / `average_sizes` / `framing` and `its_native.bench` / `memory` are the
functions behind every table above. The thorough tier (`BCIR_THOROUGH=1`) adds the ASan/UBSan
differential and a bounded libFuzzer campaign over all five generated decoders.

## 11. Next

* An **APER backend** in `cgen`: octet-aligned key and signature runs at 102.0 octets average,
  the likely point that is smaller than binary and close to COER's speed.
* **Bounds checks hoisted** over fixed-size runs in the generated encoders and decoders.
* **ECN `self-delimiting-values`** for `TO BITS` mappings, then V1.1.1's rules as an X.692
  encoding object set held byte-for-byte to the byte rule.
* The same study on the current TS 103 097 schema.
* **INTEGER value constraints on the Python OER rail.** It checks none on either side
  (`encode_oer` writes 9 for an `INTEGER (0..7)`), where the generated codecs and the PER rail
  refuse it. No ITS schema has such a constraint, so no number here depends on it; the rails
  should still agree.
