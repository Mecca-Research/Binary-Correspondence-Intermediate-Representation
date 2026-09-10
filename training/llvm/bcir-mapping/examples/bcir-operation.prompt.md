# BCIR operation source prompt

Lower a BCIR operation to a runtime-call boundary:

```bcir
op add_i32 %dst, %lhs, %rhs
```

The wrapper should keep the ABI narrow: load any local constants or flags in
LLVM IR, call one runtime function with explicit pointer/value arguments, and
return the runtime status code.
