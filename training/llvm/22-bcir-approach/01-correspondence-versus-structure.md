# Correspondence Versus Structure

## Key takeaways

- LLVM IR records **what to compute**. A correspondence IR additionally records
  **what was claimed** and **which realization was selected for it** — and keeps
  the link between them.
- The lane width in a lowered kernel is not a property of the program. The same
  BCIR module chooses a different width on each substrate, and a different one
  again when the live state Θ says the machine is hot.
- Facts LLVM *infers* — aliasing, trip counts, alignment — BCIR *declares* and
  checks. That is why a BCIR plan can be refused, and an LLVM module can only be
  malformed.
- A module that assembles and verifies proves its structure. It proves nothing
  about the legality of the plan that produced it, because the plan is not in it.

## The question the reader arrives with

Read this and nothing is surprising:

```llvm
%vc = fadd <16 x float> %va, %vb
```

Now ask two questions a compiler engineer asks by reflex, and notice that the IR
answers neither.

**Where did `16` come from?** In LLVM, from a vectorizer's cost heuristic,
consulted inside a pass, discarded once the pass ends. In BCIR, from an argmin
over an enumerated candidate set, priced by a cost model whose inputs — the
substrate profile `H`, the live state `Θ`, the policy — are all named objects
that outlive the decision.

**Where would `noalias` come from?** In LLVM, from alias analysis, which is
allowed to fail conservatively. In BCIR, from the claim's declared hazard
contract: the program *asserts* the property, and a verifier law refuses the plan
if the assertion cannot be honoured.

That difference — inferred versus declared-and-checked — is what "correspondence"
means, and everything else in this subject follows from it.

## The same program, four substrates

`bcir.examples.vector_add(1024)` is one module. It is not rewritten between the
rows below; only `H`, the substrate profile, changes.

<!-- generated: substrates -->
| substrate | lane widths | chosen | K_BCIR score |
| --- | --- | --- | --- |
| `x86-64-avx512` | 1, 8, 16 | `vec16` | 7808 |
| `x86-64-avx2` | 1, 8 | `vec8` | 9472 |
| `aarch64-neon` | 1, 4 | `vec4` | 12800 |
| `nvptx` | 1, 32 | `vec32` | 6976 |
<!-- /generated -->

Nothing in the program says `16`, `8`, `4` or `32`. Each row is
`argmin over Legal(G, H)` for its own `H`: the widths a substrate admits are
`H.lane_widths`, and the score is what `K_BCIR` charges for the winner. The nvptx
row is the useful one to sit with — it is the *cheapest* of the four, which is
exactly the kind of statement a cost model exists to make and a heuristic inside
a vectorizer cannot.

## The same program, two live states

`Θ` is the machine's live state — thermal, power, memory pressure, contention,
wear. It is not part of the program either, and it changes the answer:

<!-- generated: theta -->
| Theta | chosen | K_BCIR score |
| --- | --- | --- |
| cool (all zero) | `vec16` | 7808 |
| thermal=900, power=900 | `vec8` | 67072 |
<!-- /generated -->

A hot machine gets a narrower vector and a much larger score. The score rises
because Θ re-weights the axes (`bcir/kbcir/weights.py` adds
`theta.thermal // 20` to the thermal weight, and similarly for power,
reliability, memory, fabric and contention), so the *same* resource costs more
when the machine is already hot. The width narrows because, under those weights,
the wide candidate stops being the argmin.

This is the two-truth discipline in miniature: measured state may **rank** plans,
and may never **legalize** one. `bcir/tests/test_hot_cold.py` keeps the planning
and learning modules out of the executor's import graph so that ordering cannot
quietly invert.

## What the lowered module keeps, and what it loses

`bcir.lower.llvm.emit_kernel_ll` renders a selected realization as ordinary
opaque-pointer LLVM IR — assembles with `llvm-as`, passes `opt -passes=verify`,
and is exactly as checkable as any other module in this corpus.

What does not survive the rendering: the claim, the candidate set that was
considered and rejected, `H`, `Θ`, the policy, the resource vector, and the
legality verdict. The `.ll` is a *rendering of the realization*, and the
correspondence — claim ↔ realization — lives in BCIR, not in the file.

That is the practical reason this subject exists. A reader who audits only the
lowered IR is auditing the last step of a decision whose inputs they never saw.

## Pitfalls checklist

- Do not read a passing `llvm-as` as evidence that a plan was legal. Different
  objects, different checks.
- Do not treat the chosen lane width as a property of the program; it is a
  property of the program *under* `H` and `Θ`.
- Do not carry MLIR's sense of "legal" (may this op remain after a conversion)
  into this subject; see the fence in [`README.md`](README.md).
- Do not assume every axis of the cost vector is live. Which ones anything
  actually produces is measured in
  [`02-what-k-bcir-prices.md`](02-what-k-bcir-prices.md), and the answer is not
  "all twelve".

## Checks

[`../tools/verify-bcir-approach.py`](../tools/verify-bcir-approach.py)
regenerates both tables above from `bcir/` and compares them byte for byte.
