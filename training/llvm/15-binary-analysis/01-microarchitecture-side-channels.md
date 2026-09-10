# Microarchitecture and Side-Channel Review

LLVM IR describes program semantics, not every timing behavior of the CPU that
will eventually run the binary. A cryptographic function can be mathematically
correct in IR and still leak secrets after code generation through cache access,
branch prediction, variable-latency instructions, or layout-sensitive effects.

## What static IR can and cannot prove

Static IR can identify suspicious patterns:

- branches whose condition depends on a secret value;
- table lookups where a secret influences the address;
- early exits that reveal how much of a secret matched;
- calls to helpers whose constant-time contract is unknown;
- target-specific intrinsics or inline assembly with variable latency.

Static IR cannot, by itself, prove hardware constant time. It does not model
cache replacement, branch predictor state, speculative execution, turbo/frequency
changes, TLB behavior, or code-layout effects introduced after instruction
selection and linking.

## Common side-channel review questions

| Question | Static signal | Dynamic signal |
| --- | --- | --- |
| Does a secret change control flow? | `br i1 %secret_dependent` or `switch` on secret-derived values. | Branch-miss counts or trace paths vary with secret classes. |
| Does a secret change memory addresses? | GEP/load address uses secret-derived index. | Cache misses, TLB misses, or load latency distributions vary with secret classes. |
| Does code use variable-latency operations? | `sdiv`, `urem`, target intrinsics, calls, or inline assembly near secrets. | Cycle distributions vary despite identical public inputs. |
| Did backend layout alter behavior? | Hot/cold block layout, jump tables, outlined helpers. | I-cache misses or branch target buffer behavior changes across builds. |

## IR patterns to distrust around secrets

```llvm
; Secret-dependent branch: likely not constant time.
define i32 @branch_on_secret(i32 %secret, i32 %a, i32 %b) {
entry:
  %bit = and i32 %secret, 1
  %cond = icmp ne i32 %bit, 0
  br i1 %cond, label %then, label %else

then:
  ret i32 %a

else:
  ret i32 %b
}

; A select is a better starting point, but still requires backend review.
define i32 @select_on_secret(i32 %secret, i32 %a, i32 %b) {
entry:
  %bit = and i32 %secret, 1
  %cond = icmp ne i32 %bit, 0
  %v = select i1 %cond, i32 %a, i32 %b
  ret i32 %v
}
```

Even the `select` version is not a proof. A target may lower a select to a branch
for some types or cost models, so inspect generated assembly and measure.

## Constant-time review loop

1. **Classify inputs** as public or secret before reading IR.
2. **Mark taint roots**: secret parameters, loads from secret buffers, and return
   values from secret-producing helpers.
3. **Trace taint to control flow and addresses**: branch conditions, switch keys,
   GEP indices, load/store pointers, and call arguments.
4. **Compile representative targets** with the same optimization, LTO, and PGO
   settings used in production.
5. **Inspect assembly** for secret-dependent branches, table lookups, calls, or
   variable-latency instructions.
6. **Measure paired inputs** with the same public input and different secret
   classes. Compare cycles, branch misses, cache misses, and trace paths.
7. **Treat disagreement as a finding**: if static and dynamic evidence disagree,
   explain why before declaring code constant time.

## BCIR/BCSA implications

For binary-code similarity, side-channel-sensitive functions should not be
matched only by opcode or CFG isomorphism. Two binaries can implement the same
mathematical transform while one uses a secret-dependent table and the other uses
bitslicing. Add dynamic evidence and target metadata to the comparison key when
security properties matter.
