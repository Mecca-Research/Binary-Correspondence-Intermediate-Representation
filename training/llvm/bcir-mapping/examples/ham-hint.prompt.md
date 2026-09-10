# HAM hint source prompt

Lower this advisory memory hint:

```bcir
ham.prefetch resource(%input) offset(%byte_offset) locality(3) read
ham.provenance "tile:7 lane-group:0"
```

The lowered IR should compute the byte address explicitly, call
`llvm.prefetch.p0`, and carry non-semantic BCIR provenance metadata.
