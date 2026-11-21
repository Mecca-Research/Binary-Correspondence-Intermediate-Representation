"""PORT-8's one native-ready piece, measured: the hard-fact filter beside the native Q15 top-k."""

import hashlib, json, random, statistics, sys, time
from bcir.kbcir.native_ai import NativeAIKernels, NativeOptimizationIndex
from bcir.kbcir.optimization_memory import (
    OptimizationFact,
    OptimizationMemory,
    OptimizationPattern,
    _prepare_optimization_query,
)

r = random.Random(1)
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
kernels = NativeAIKernels.build(sys.argv[1])
rows = []
for n, dim, nf in ((256, 64, 16), (1024, 128, 32), (4096, 128, 64)):
    facts = tuple(sorted(OptimizationFact(f"s{k}", "p", str(k)) for k in range(nf)))
    pats = tuple(
        sorted(
            (
                OptimizationPattern(
                    h(f"p{i}"),
                    tuple(r.randrange(-2000, 2000) for _ in range(dim)),
                    h(f"c{i}"),
                    h(f"e{i}"),
                    tuple(sorted(r.sample(facts, 4))),
                )
                for i in range(n)
            ),
            key=lambda p: p.pattern_sha256,
        )
    )
    mem = OptimizationMemory(h("m"), h("e"), pats, facts)
    idx = NativeOptimizationIndex(kernels, mem)
    q = tuple(r.randrange(-2000, 2000) for _ in range(dim))
    active = facts[: nf // 2]
    prep, total = [], []
    for _ in range(15):
        t0 = time.perf_counter()
        _prepare_optimization_query(mem, q, top_k=8, active_facts=active)
        prep.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        idx.query(q, top_k=8, active_facts=active)
        total.append(time.perf_counter() - t0)
    rows.append(
        {
            "patterns": n,
            "dimension": dim,
            "facts": nf,
            "filter_ms": round(statistics.median(prep) * 1e3, 3),
            "query_ms": round(statistics.median(total) * 1e3, 3),
        }
    )
    print(json.dumps(rows[-1]), flush=True)
