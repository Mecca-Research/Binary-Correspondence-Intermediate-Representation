# 02 — Quantization: what MLIR's `quant` leaves open, and what BCIR pins

The `quant` dialect is three operations. It is also the cleanest place in this corpus to
see a general rule: **a specification that declines to fix a behaviour has not removed the
behaviour, it has moved the decision somewhere less visible.** Both of the things MLIR
deliberately leaves to the pipeline here are things BCIR's own quantizer decides
explicitly, and the two decisions differ.

## What a quantized type is

`!quant.uniform<i8:f32, 0.02:-1>` is an `i8` that *means* `(code − (−1)) × 0.02`. The
arithmetic is in the type, so the IR can carry a quantized tensor around without spelling
it out. `quant.qcast` and `quant.dcast` are where the meaning is spent, and
`--lower-quant-ops` is what spends it.
[`examples/quant-cast-arithmetic.mlir`](examples/quant-cast-arithmetic.mlir) lowers to:

```mlir
// quant.qcast  --  float in, code out
%0 = arith.divf %arg0, %splat : tensor<4xf32>      // x / scale
%2 = arith.addf %0, %1 : tensor<4xf32>             // + zeroPoint
%3 = arith.fptosi %2 : tensor<4xf32> to tensor<4xi8>

// quant.dcast  --  code in, float out
%1 = arith.sitofp %0 : tensor<4xi8> to tensor<4xf32>
%3 = arith.subf %1, %2 : tensor<4xf32>             // - zeroPoint
%4 = arith.mulf %3, %splat : tensor<4xf32>         // * scale
```

The dialect's third operation closes the loop: `quant.scast` is the specification's
`reinterpretCast` step, moving between a code and the quantized type that gives it meaning
without touching a bit. It is the only one of the three that is free.

No surprises in the algebra. The whole subject is in that `arith.fptosi`.

## The first open decision: rounding

`quant.qcast`'s own ODS description — read out of `QuantOps.td` in the installed MLIR, not
from a web page — says so directly:

> The numerical results produced by the algorithm above may vary depending on the rounding
> methods used by `convertIntToFloat()`, `convertFloatToInt()`, `clamp()`, division (`/`),
> and addition (`+`). This operation does not define specific rounding methods; instead, it
> is the responsibility of a transform pipeline to determine which rounding method to apply
> when this operation is broken down into lower-level dialects.

That is an honest specification. It is also a decision that has to be made by somebody, and
the stock pipeline makes it: `arith.fptosi`, which truncates **toward zero**. `2.6 / 1.0`
quantizes to `2`, not `3`. A model quantized through this path and a model quantized by a
framework that rounds to nearest will disagree on roughly half their codes by one unit, and
nothing anywhere reports it.

BCIR makes the opposite choice, and makes it in one place —
`bcir/kbcir/quantize.py`, at the repository root:

```python
def _round_half_away(x: float) -> int:
    """Deterministic round-half-away-from-zero (symmetric; NOT Python's banker's `round`)."""
    return int(math.floor(x + 0.5)) if x >= 0.0 else -int(math.floor(-x + 0.5))

def _quantize_code(x, cmax, scale, rounding):
    q = x / scale
    c = int(q) if rounding == "truncate" else _round_half_away(q)   # truncate = toward zero
    return cmax if c > cmax else -cmax if c < -cmax else c          # saturate into the lane
```

Truncation is available and is spelled `rounding="truncate"`. It is not the default, and it
is not silent.

## The second open decision: what happens out of range

`quant.qcast`'s specification includes a clamp:

```
storedValue        = convertFloatToInt(storedValueFloat, storageType)
storedValueClamped = clamp(storedValue, storageMin, storageMax)
```

The lowering emits that clamp — *sometimes*.
[`examples/quant-clamp-only-when-narrowed.mlir`](examples/quant-clamp-only-when-narrowed.mlir)
declares `i8<-100:100>`, a range narrower than `i8`'s own, and gets it:

```mlir
%3 = arith.fptosi %2 : tensor<4xf32> to tensor<4xi8>
%4 = arith.maxsi %3, %splat_1 : tensor<4xi8>
%5 = arith.minsi %4, %splat_2 : tensor<4xi8>
```

`quant-cast-arithmetic.mlir` declares the plain `i8` — storage range equal to the storage
type — and gets no `maxsi`/`minsi` at all. Which looks like a sound optimisation: clamping
an `i8` to `[-128, 127]` cannot do anything.

It is sound only if `convertFloatToInt` lands in range, and `arith.fptosi` has no such
obligation.
[`../23-version-movement/03-conversions-and-shifts.md`](../23-version-movement/03-conversions-and-shifts.md)
is the chapter on exactly this, with its own assembled fixture: an out-of-range float-to-int
conversion is **poison** — not a saturated endpoint, not a wrapped value, and with no
diagnostic. You can watch the folder decline to invent one. Quantizing `1000.0` at scale
`0.02` needs the code `50000`:

```mlir
%cst = arith.constant dense<5.000000e+04> : tensor<1xf32>
%0 = arith.fptosi %cst : tensor<1xf32> to tensor<1xi8>
return %0 : tensor<1xi8>
```

The division folded. The `fptosi` did not, because there is no value it is entitled to
produce. A pipeline that saturated would have folded this to `127`.

BCIR's line `return cmax if c > cmax else -cmax if c < -cmax else c` is that clamp, applied
unconditionally and before anything can become poison. Note also `cmax = (1 << (bits-1)) - 1`:
BCIR's 8-bit lane is symmetric, `[-127, 127]`, and does not use `-128` at all.

## The trap: a round trip that measures nothing

Quantization is lossy. That loss is the entire subject. So the natural first experiment —
quantize, dequantize, compare — is the one to be most careful with, and in MLIR 23.1.1 it
does not work:

```console
$ mlir-opt quant-round-trip-folds-away.mlir --canonicalize
  func.func @round_trip(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }
```

The canonicalizer folds `dcast(qcast(%x))` to `%x`. The measurement reports exactly zero
error, and it will keep reporting zero however the scale, the zero point or the rounding
mode are changed, because there is nothing left to measure. This is the general hazard in
its purest form: *a round trip passes when both halves share the same assumption*.

[`examples/quant-round-trip-folds-away.mlir`](examples/quant-round-trip-folds-away.mlir)
pins the fold, so this warning cannot outlive the behaviour it describes.

## The comparison, in one table

Every cell is read out of a source this repository can point at — MLIR's `QuantOps.td` and
the `--lower-quant-ops` output on one side, `bcir/kbcir/quantize.py` on the other.

| | MLIR `quant` (23.1.1, stock lowering) | BCIR `kbcir/quantize.py` |
| --- | --- | --- |
| Rounding | undefined by the op; the pipeline picks `arith.fptosi`, toward zero | round-half-away-from-zero; `truncate` is an explicit opt-in |
| Out of range | clamped only when the declared storage range is narrower than the storage type; otherwise `arith.fptosi`, which is poison | saturated to `±cmax`, always |
| Zero point | affine, carried in the type | none; the range is symmetric |
| Code range | the storage type's, e.g. `[-128, 127]` for `i8` | `[-cmax, cmax]` with `cmax = 2^(bits-1) − 1`, so `[-127, 127]` |
| Scale | an arbitrary `f32` in the type | an exactly chosen power of two |

None of these make MLIR wrong. `quant` is a *representation*, deliberately leaving policy to
the pipeline that lowers it; BCIR is a *plan* that has to be reproducible bit-for-bit across
two rails, so leaving the rounding mode open is not available to it. The reason to know both
columns is that porting a quantized model across them is not a format conversion: the
numbers change, and the two places they change are rounding and the range edge.

## Pitfalls checklist

- Do not assume a quantization round trip loses anything in an IR the canonicalizer has
  seen. Check for the conversion ops before trusting an error measurement.
- Do not read "the operation does not define a rounding method" as "rounding does not
  matter here". It means the decision is downstream and unlabelled.
- Do not rely on the specification's `clamp` step at the storage type's full range: the
  lowering omits it, and the conversion it relies on is poison out of range.
- Do not port codes between BCIR and MLIR `quant` by reinterpreting bits. The zero point,
  the range edge and the rounding rule all differ.
- Do not use `-128` when matching BCIR's 8-bit lane; it is symmetric by construction.

## Checks

All three fixtures are Tier 3 entries in
[`../autograder/mlir-examples.json`](../autograder/mlir-examples.json).

- [`examples/quant-cast-arithmetic.mlir`](examples/quant-cast-arithmetic.mlir) must produce
  `arith.divf`, `arith.addf`, `arith.fptosi`, `arith.sitofp`, `arith.subf` and `arith.mulf`,
  and must contain **no** `arith.maxsi` or `arith.minsi`. That forbid pair is the load-bearing
  half: it is the assertion that the full storage range gets no clamp.
- [`examples/quant-clamp-only-when-narrowed.mlir`](examples/quant-clamp-only-when-narrowed.mlir)
  is the same lowering over a narrowed range and must produce both of them. The two fixtures
  only mean something together — either alone would be consistent with the clamp being
  unconditional, or with it being absent.
- [`examples/quant-round-trip-folds-away.mlir`](examples/quant-round-trip-folds-away.mlir)
  must canonicalize to `return %arg0` with **none** of the quantization arithmetic left —
  no `quant.*` cast, and no `arith.divf`, `addf`, `fptosi`, `sitofp`, `subf` or `mulf`. The
  long forbid list is not thoroughness for its own sake. `return %arg0` alone is too weak to
  pin this: `--lower-quant-ops` folds the `dcast` as well, so its output returns `%arg0`
  too, and a fixture requiring only that would pass under a pipeline that does not
  demonstrate the hazard at all. The claim is that nothing survives, so the check has to be
  that nothing survives.

BCIR's half of the table is not re-derived here: its rounding rule is already gated by
[`../../tools/verify_embeddings.py`](../../tools/verify_embeddings.py), which checks the
corpus's own quantizer against `bcir.kbcir.quantize._round_half_away` wherever BCIR is
importable, and against half-integer inputs — the only values where round-half-away and
Python's banker's rounding disagree — where it is not.
