# `bcir/` — the executable conformance oracle

A dependency-free, deterministic Python realization of BCIR + K_BCIR + GEM. It is
the runnable reference for the MLIR law under [`../mlir/`](../mlir/) (see
[`../docs/PARITY.md`](../docs/PARITY.md)) and demonstrates LangRef Milestones 5–7
today, on a host with only `python3` and `clang`.

## Layout

```
model/    BCIR-0..2 semantic model: lanes, opcodes, resources, claims, phases
kbcir/    K_BCIR (BCIR-3): cost vectors, target profiles + memory hierarchy,
          min-plus semiring, Theta-driven weights, the realization optimizer
gem/      GEM (BCIR-4): StreamPack hydration with provenance + generation tags
lower/    BCIR-5: legal LLVM IR emission that clang compiles + runs + self-checks
verify/   runnable subset of LangRef verifier laws R1-R12
examples.py  the goal-graph corpus (vector_add, saxpy_strided, histogram_gather, ...)
run.py    the CLI
tests/    17 checks + a dependency-free runner
```

## Run it

```bash
# K_BCIR plan for a program on a target under a runtime state Theta:
python -m bcir.run vector_add --target x86_avx512 --theta cool
python -m bcir.run vector_add --target nvidia_ptx           # GPU warp -> vec32
python -m bcir.run vector_add --target x86_avx512 --theta hot  # replans vec16 -> vec8

# Lower + compile + run the selected kernel via clang (self-checking):
python -m bcir.run vector_add --target x86_avx512 --run

# Tests (no pytest required; also works under `python -m pytest bcir/tests`):
python -m bcir.tests.run_all
```

## The central equation, made runnable

```
K_BCIR(G | H, Theta) = min_pi  sum_i  T_i (X) f_i(pi)  =  min_pi C_H(pi, Theta)
```

`optimize()` builds a layered realization DAG (one column of legal candidate
lowerings per claim), couples each base cost `T_i` by the path-context factor
`f_i(pi)`, scalarizes under `w(H, Theta, phase, policy)`, and runs a tropical
(min,+) shortest path to select `pi*`. Everything is integer/Q-fixed for
determinism — the worked example (`vector_add`, AVX-512, cool Theta) scores
exactly **7808** and selects `vec16`, matching `mlir/examples/full_vec_add_ct1.mlir`.
