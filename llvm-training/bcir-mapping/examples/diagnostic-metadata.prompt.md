# Diagnostic metadata source prompt

Preserve source diagnostic tags while lowering this operation:

```bcir
claim %c42 at graph.edge[3] provenance("rule:resource-bounds") {
  read = [%r0]
  op = load_i32
}
```

Attach metadata to the lowered GEP/load so verifier-valid IR still carries the
claim, graph, and rule names needed by diagnostics after optimization.
