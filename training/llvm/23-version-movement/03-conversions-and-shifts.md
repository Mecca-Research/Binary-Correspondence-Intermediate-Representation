# The casts and shifts the corpus listed but never ran

Nine instruction opcodes appear in this corpus only as entries in the syntax table in
[`../01-syntax/02-instruction-format.md`](../01-syntax/02-instruction-format.md): `ashr`,
`fneg`, `fpext`, `fptrunc`, `fptosi`, `fptoui`, `sitofp`, `uitofp`. Named, never
demonstrated — and four of them have behaviour that surprises people who assume the
obvious.

Everything asserted below was produced by LLVM 23's constant folder (`opt -O2 -S`) over
[`examples/conversions-and-shifts.ll`](examples/conversions-and-shifts.ll), not written
from memory.

## `ashr` and `lshr` differ only in what fills the top

```llvm
ashr i32 -8, 1    ; folds to  -4
lshr i32 -8, 1    ; folds to  2147483644
```

`ashr` shifts in copies of the sign bit, so a negative number stays negative and the
result is a division by two rounding toward negative infinity. `lshr` shifts in zeros, so
the sign bit becomes an ordinary value bit and `-8` reads as a large positive number.

There is one shift-left, `shl`, because filling the low bits with zeros is the only
sensible choice. The asymmetry in the names is a real asymmetry in the operation.

**The rule:** the signedness lives in the *instruction*, not in the type. `i32` is
neither signed nor unsigned in LLVM; `ashr` versus `lshr` is where you say which one you
meant.

## Float to int is `poison` out of range

This is the one that costs people days.

```llvm
fptosi float 4294967296.0 to i32   ; folds to  poison
fptoui float -1.0 to i32           ; folds to  poison
```

Not saturation. Not zero. Not a trap. **Poison**, which propagates through everything it
touches and can be materialised as any value at all — so a single out-of-range conversion
turns a later, apparently unrelated, computation into nonsense with no diagnostic
anywhere.

The second case is worth staring at: `-1.0` is a perfectly ordinary float. It is poison
only because the *destination* is unsigned. Signedness is in the opcode, again.

When the input is not provably in range, the saturating intrinsics are the answer, and
they exist precisely because the instructions behave this way:

```llvm
%r = call i32 @llvm.fptosi.sat.i32.f32(float %f)   ; clamps; defined for every input
```

`llvm.fptosi.sat` / `llvm.fptoui.sat` clamp to the destination's range and map NaN to
zero. C's `(int)someFloat` is the instruction, not the intrinsic — which is why that cast
is undefined behaviour in C for out-of-range values, and why the two facts are the same
fact seen from two levels.

## Int to float loses precision silently

```llvm
sitofp i64 9007199254740993 to double   ; folds to 0x4340000000000000
```

That bit pattern is 9007199254740992 — the input minus one. A `double` has 53 significand
bits, so 2^53 + 1 is not representable and the conversion rounds. Nothing signals it.

This is not an edge case in practice: any 64-bit identifier, hash, or file offset above
2^53 loses its low bits on the way through a `double`. If a value must survive the round
trip, it must not go through floating point.

`uitofp` differs from `sitofp` only in how it reads the source bits — same trap, different
interpretation of the input.

## `fpext` is exact, `fptrunc` is a rounding operation

`fpext float -> double` is always exact: every float is a double. `fptrunc double ->
float` may round, which is why it is a separate opcode and not a `bitcast`. A `bitcast`
would reinterpret the bits and produce a completely different number; these two convert
the *value*.

## `fneg` is not `fsub -0.0, x`

`fneg` flips the sign bit and does nothing else. The older spelling `fsub float -0.0, %f`
is a subtraction and follows subtraction's rules — which differ for signed zeros and for
NaN payloads. If you mean negation, write `fneg`; it is also the form the optimiser
recognises.

## Pitfalls checklist

- Do not read signedness off the type. `i32` has none; `ashr`/`lshr`, `fptosi`/`fptoui`
  and `sitofp`/`uitofp` are where it is stated.
- Do not assume an out-of-range float-to-int conversion saturates or wraps. It is poison —
  reach for `llvm.fptosi.sat` / `llvm.fptoui.sat` when the range is not provable.
- Do not route a large integer through `double` and expect it back. Above 2^53 it will not
  come back.
- Do not use `bitcast` to change a float's width, or `fptrunc`/`fpext` to reinterpret bits.
  They are different operations with different results.
- Do not write `fsub -0.0, x` for negation in new IR.

## Checks

[`examples/conversions-and-shifts.ll`](examples/conversions-and-shifts.ll) is assembled
and verified by [`../tools/verify-examples.sh`](../tools/verify-examples.sh) on every
host, including the corpus baseline — nothing in it is newer than LLVM 15. Its constants
are chosen to be exactly representable for that reason: LLVM 15 and 18 reject a decimal
float literal that is not exact in its type, where LLVM 23 accepts it, so `4294967296.0`
assembles everywhere and `1.0e30` would not.
