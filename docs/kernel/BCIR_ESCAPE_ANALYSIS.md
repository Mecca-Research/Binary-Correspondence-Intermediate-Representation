# Escape analysis, indirect-call narrowing and the effect footprint of a C unit (GEM+ G10)

`CompileResult.commute(a, b)` answers a scheduling question: may the cfront frontend reorder
two functions of a unit? It is only as good as each function's **footprint**, the memory it
and everything it calls read and write. Before G10 the footprint took a function's writes from its
claims' `wr` operands. But a cfront store is `c.store rd=(base, [index,] value) wr=()`, so every
store was recorded as a read. Writes through a pointer, writes to a static local and writes to
heap memory were missing, and the call graph had no edge for an indirect call. Measured on the
parent, 47 pairs of functions the dynamic witness shows diverging were reported as commuting. Both
rails were wrong the same way, and they disagreed on 20 units besides.

G10 replaces that with one points-to analysis over the whole unit. The same analysis answers two
questions the roadmap asks of it (GEM+ roadmap G10, staged plan S5-B):

- which local arrays never leave their function (**escape**): a local that never crosses a call
  boundary can be a frame slot rather than a registry resource;
- which functions an indirect call can reach (**narrowing**): a site narrowed to known targets is a
  call-graph edge, and its footprint is its targets' footprint, not "anything".

| Module | Holds |
|---|---|
| [`bcir/frontends/cfront/escape.py`](../../bcir/frontends/cfront/escape.py) | the analysis (`analyze`), its answers (`EscapeResult`: `footprints`, `objects`, `candidates`, `sites`, `edges`) and the two reports both rails print |
| [`runtime/c/bcir_cfront.c`](../../runtime/c/bcir_cfront.c) (`esc_*`) | the C twin: `bcir_cfront_effects` / `bcir_cfront_escape`, `bcir-cc --emit-effects` / `--emit-escape` |
| [`bcir/frontends/cfront/pipeline.py`](../../bcir/frontends/cfront/pipeline.py) | `CompileResult.escape`, `.effects` (the footprints) and `.commute` |
| [`bcir/tests/escape_fixtures.py`](../../bcir/tests/escape_fixtures.py), [`tools/perf/check_escape.py`](../../tools/perf/check_escape.py) | the generator, the witness, the corpus, the grader and the gate |

## The analysis

The analysis is Andersen's: inclusion-based, and insensitive to flow, calling context and fields. It
works in an **open world**. A translation unit is not a whole program: another unit can call a
non-static function with anything, write a global, or receive anything passed to it.

**Objects.** An object is one of the following:

- a file-scope global (`G`, by name);
- a static local (`S`, named `function.name`);
- an automatic local or temporary (`L`, function and rid);
- a function's **heap** (`H`, every allocation it makes);
- a function (`F`);
- the string literals (`STR`, read-only);
- unknown memory (`TOP`).

Every storage object has a points-to set, `pt`: the objects the pointers stored in it may point
to. What an object *holds* is its `pt`, plus `TOP` when code the unit cannot see can have written
it: a global, a static, or an object that escaped.

**The C type decides the reading.** How a rid is read depends on its C type:

- An array used as a value is its own address (the decay).
- A struct, a pointer or a scalar used as a value is what it holds.
- A load or store touches an array, a struct or a declared scalar variable **in place** (`a++` on a
  parameter stores to `a`). Through a pointer, or a temporary holding an address, it touches what
  that holds.

The C twin reads the kind from what it declares:

- a parameter's `bcir_ctype.kind`;
- `is_array` and `is_vla` on a resource. `is_array` is a flag this slice adds, so that `T a[1]` is
  still an array;
- `is_pointer`, the other flag this slice adds. The twin models a file-scope pointer's slot as a
  scalar resource, so without the flag `gp[i] = v` would write `gp` rather than what `gp` points
  to.

**Rules**, per claim:

| Claim | Constraint |
|---|---|
| `c.load` | the result holds what the touched objects hold |
| `c.store` | each touched object holds what the value points to |
| `c.addrof` | the result points to the base object (and, through a pointer, what it holds) |
| a direct call to a function of the unit | each formal holds what its actual points to; each result holds what the callee returns; a variadic extra goes to unknown memory |
| a call the unit cannot see (`c.call.tu:`, `c.call.extern:`, an undefined callee, inline asm) | the actuals **escape** into unknown memory; the results hold `TOP` |
| an indirect call | binds every function of the unit the pointer may hold; goes external as well when it may hold `TOP` or another unit's function |
| a library routine (`libm`, builtins, `va_*`) | keeps no pointer, and returns what it was given, copies between its operands included |
| an allocator (`malloc`, `calloc`, `realloc`, `aligned_alloc`) | the result points to the caller's heap object `H` |
| another library result of pointer type | the result also holds `TOP` |
| an atomic | reads and writes, and exchanges pointers between, what its operands touch |
| `(T *)v`, a pointer cast | holds what `v` points to, and `TOP` unless `v` is provably a pointer or the null constant (see below) |
| every other claim | the result holds what its operands point to |

A pointer cast's operand is *provably a pointer* when it is one of these:

- a pointer or array variable;
- an address (`&x`);
- another pointer cast;
- a function;
- a string literal.

Anything else may be an integer: `(T *)0x40000000u` is a device register, and `(T *)addr` is
wherever the integer came from.

**Seeds.** What the unit cannot see is where the open world enters the analysis:

- Every non-scalar parameter of a function another unit can call holds `TOP`. Such a function is one
  that is not `static`, or whose address is taken, either by a claim or **by a file-scope
  initializer** (an ops table `struct ops t = { handler };`: whoever reads `t` can call `handler`).
- A scalar parameter becomes a pointer only through a cast, and the cast rule already makes that
  cast `TOP`.

**Monotone, so order-free.** Every rule only grows the points-to sets. An indirect call binds each
known target as soon as the pointer may hold it, and goes external as soon as the pointer may hold
something unknown. It does not decide once from a partial set: a site first seen empty used to go
external, and its arguments escaped for good. So the fixpoint is the least one, and it does not
depend on the order the claims are visited in. The two rails order sibling expressions differently,
so byte-identical reports need this property. `test_the_answer_does_not_depend_on_the_order_the_claims_are_visited_in`
shuffles every function's claims and requires the same answer.

## The answers

**Escape.** A named automatic local memory object gets one of three verdicts:

- **escaping**: reachable from what escaped, from what a global or static holds, or from its own
  function's return values;
- **lent**: not escaping, but reachable from some call's actuals, so it crosses a call boundary;
- **nonescaping**: neither.

The roadmap's candidates are the declared-extent local arrays. The corpus has 39 of them, and all 39
are proved nonescaping (`escape.unproved` 39 → 0).

**Narrowing.** Each indirect site (`c.call.indirect`, `c.call.imember`) gets one of three answers:

- **resolved**: exactly one function of the unit;
- **known**: a set of them;
- **unknown**, printed `*`: the pointer may hold `TOP` or another unit's function, or holds nothing
  known.

Resolved and known sites extend the call graph (`EscapeResult.edges`). The cfront corpus has 22
sites: 4 resolved, 7 known. The other 15 are at the open world's floor. Each calls a function
pointer, or a member of a struct, that an exported function takes as a parameter, so another unit
may pass anything and no sound analysis narrows them (`icall.unknown` 22 → 15, its floor). Of the
7 known sites, `cond_local_fp`'s pointer really holds either of two functions. `use_local_fp`
reassigns its pointer between two calls, so only a flow-sensitive analysis could resolve its two
sites (`icall.unresolved` 22 → 18, against a floor of 16).

**Footprints.** A function's reads and writes, by name, are its own accesses and those of every
function it can reach through the call graph, narrowed edges included:

- a global is named by its name;
- a static local by `function.name`;
- unknown memory, and a local or heap object another activation can see, is `*`;
- a local or heap object of one of those activations that nothing outside can reach is private, and
  is not listed.

A **device access** reads and writes `*`. Device state is observable, and two reads of a FIFO
register do not commute. A claim is a device access when it is MMIO-domain (`_device`,
`esc_device`). R3's pass, the last step of both lowerings, makes every claim that reads or writes a
device region MMIO-domain, so the domain carries the **base resource**, and how a lowering spelled
the access does not decide it. The analysis once also read a load's base resource itself, because
the twin lowered `p[i]` through a `volatile T *` as an ordinary load. R3 on both rails left no input
that reaches that second reading, so it was removed, and the base rule lives in R3 alone.

Both frontends carry `volatile` to every place a C program puts it: a pointer to volatile (a
parameter, a local, a file-scope pointer, a struct member, an array element, a cast), a volatile
member or member array, a member of a volatile struct, a volatile file-scope or automatic object or
array, and a volatile static. A resource whose type holds volatile storage, or points at it, is an
MMIO resource on both rails, and every access through it is a device access -- so two readers of a
volatile global, of a file-scope pointer to volatile, or of a register block through a member
pointer do not commute. The frontends' volatile corpus (`bcir/tests/volatile_fixtures.py`) holds
each place against each access form at eight element widths.

Two footprints conflict when one writes what the other reads or writes. `*` conflicts with any
name, and an empty footprint commutes with everything.

## The reports (both rails, byte for byte)

`bcir-cc --emit-effects` and `escape.effects_report` print one line per function in unit order,
then the commute matrix over every pair:

```
fn=pw reads=- writes=g_buf
fn=pr reads=g_buf writes=-
fn=next_id reads=next_id.n writes=next_id.n
commute pw pr = 0
```

`bcir-cc --emit-escape` and `escape.escape_report` print each function's named locals by verdict,
then its indirect sites' target sets. Each set is comma-joined, the sets are sorted and joined by
`;`, and `*` means unknown:

```
fn=f nonescaping=priv lent=lent escaping=esc icalls=-
fn=two nonescaping=- lent=- escaping=- icalls=add1,dbl
```

The C twin keeps six operands per claim and drops the rest, so a call with more is **refused** on
both rails. The unit's footprints are all `*`, no local is proved private, and no site is narrowed.
The escape report reads `truncated=1` followed by `fn=<name> refused`. The twin marks the dropped
operand on the claim (`bcir_claim.truncated`), so it sees what it cannot hold.

## What is not claimed

- **Flow, context and field sensitivity.** `f = add1; f(x); f = dbl; f(x)` narrows both sites to
  `{add1, dbl}`; a struct of two function pointers holds both.
- **Global initializers' pointer contents.** A pointer loaded from a global is unknown, because
  another unit can write it.
- **Type punning.** An integer stored into memory and read back as a pointer through a union is not
  modelled.
- **The heap is one object per allocating function.** Two buffers one function allocates are not
  told apart.
- **The dynamic witness cannot observe a device.** The MMIO rule is held by the parity rows and the
  tests, not by the witness.

## Evidence

The rows live in group `escape` (`tools/perf/gemplus_baseline.py --group escape`). They are
measured by `escape_fixtures.measure`, which the tests and the gate share. RED is the parent,
8d3aab84, judged by the same fixtures.

| Row | RED | GREEN |
|---|---|---|
| `escape.unproved` (of 39 candidates, floor 0) | 39 | 0, at its floor |
| `icall.unknown` (of 22 sites, floor 15) | 22 | 15, at its floor |
| `icall.unresolved` (of 22 sites, floor 16) | 22 | 18 |
| `escape.verdict.mismatch` (73 locals of 24 generated units) | 73 | 0 |
| `icall.target.mismatch` (20 sites) | 20 | 0 |
| `effects.commute.unsound` (1,867 witnessed pairs of the corpus and the generated units) | 119 | 0 |
| `effects.parity.mismatch` (200 units: corpus, generated, forms) | 24 | 0 |
| `escape.parity.mismatch` (200 units) | 200 | 0 |

The corpus rows count what is not yet proved or narrowed, so each falls toward a proved floor.
Because the analysis is sound, a value under a floor is itself a defect, and
`tools/perf/check_escape.py` holds both sides: a fault that narrows the open-world sites to nothing
fires `icall.unknown` under its floor.

Each of the checks below is held to the same standard.

- **The witness.** It is one C program per unit, with `main(pair, order)`. Each order runs in its
  own process, over shared buffers (one per pointee type, so pointer parameters alias). It hashes
  every result, buffer and global, and hashes integers by value (a `_BitInt`'s padding bits are
  unspecified). It decides 1,867 pairs: 846 of the corpus and 1,021 of the generated units. A
  static function is driven only when a file-scope initializer hands it out; the witness reads that
  from the AST, not from the lowering it judges.
- **The generator.** `generate(seed)` builds units whose verdicts and targets are known by
  construction. It mixes in heap buffers, a pointer made from an integer, a static writer, and a
  callback in a file-scope ops table. Each unit ends with two exported functions that differ only
  through a static local of a helper they both call. That pair holds the static-local rule to the
  witness, not only to parity.
- **The forms.** `form_units()` holds 476 small functions in four units, one unit per storage place
  (file scope, static, automatic, parameter). They cover 15 declaration kinds, the volatile ones
  included, against 27 access forms. The forms were chosen by sweeping every combination and keeping
  those that are C a compiler accepts and both rails lower. That sweep found the twin touching a
  file-scope pointer in place (12 forms) and missing the device access through a volatile pointer
  (30 forms). A second sweep, once volatile reached globals and members on both rails, added 90:
  stores through a file-scope pointer to volatile and into a volatile file-scope array, reads and
  writes of a volatile scalar, `*a` of an array, `*g` through a file-scope pointer, and a struct
  whose member points at volatile storage. Array parameters are pointers, so the parameter place
  leaves them out; a member of a file-scope struct and `**` through a file-scope pointer are forms
  the twin does not lower. The list is pinned, so a frontend that stops lowering a form fails its
  unit.
- **Refusals count.** A unit the twin refuses is a parity failure, except the one pinned
  preprocessor limit (`cfront_sec_cppmacro.c`). A twin that reported on nothing used to pass both
  parity rows.
- **Rail parity beyond the rows.** The rows compare 200 units. On top of those, 300 programs from
  the rich `tools/c/fuzz_cfront.py` generator produced 600 reports, byte-identical on both rails.
- **Memory safety of the twin's analysis.** It is clean under ASan and UBSan over every fixture,
  generated unit and form, and `clang --analyze` reports nothing. The memory-discipline gate's
  allocation-fault injection covers it: every failure leaves an empty report and no leak.
- **Faults.** [`tools/testing/faults/escape.json`](../../tools/testing/faults/escape.json)
  injects 27 defects into the oracle, the lowering, the twin and its driver, and each one is caught
  by the row it names. Two of them leave R3's pass undone, one per rail, and the effect parity row
  sees the device accesses go. One of them makes every report of the twin fail: the parity rows
  fire, where they used to read 0.

**Cost.** The analysis takes about 5% of `compile_unit`'s time over the corpus (0.09 s of
1.7 s), and 0.09 s on the 7,630-claim scale unit. It runs eagerly because the verified-C
attestation reads the footprints. The C twin computes nothing unless asked.
