"""Min-plus (tropical) shortest path over the realization DAG.

The realization DAG is built in layered topological order: node 0 is SOURCE, then
one column of candidate nodes per claim, then a SINK. Every edge runs from a
lower node id to a higher one, so node-id order *is* a topological order and a
single relaxation sweep computes the min-plus shortest path. Edge weights are
scalarized K_BCIR costs (>= 0), so this also agrees with Dijkstra.

This is the `semiring = #bcir.semiring<min_plus>` of the LangRef: composition is
(+) along a path and (min) across alternative paths.
"""

from __future__ import annotations

INF = float("inf")


def dag_shortest_path(
    n: int,
    adj: list[list[tuple[int, int]]],
    source: int = 0,
) -> tuple[list[float], list[int]]:
    """Return (dist, pred) for a DAG whose edges only go to higher node ids.

    `adj[u]` is a list of `(v, weight)` out-edges. `dist[v]` is the minimum total
    weight from `source` to `v`; `pred[v]` is the predecessor on that path (-1 if
    unreached or the source).
    """
    dist: list[float] = [INF] * n
    pred: list[int] = [-1] * n
    dist[source] = 0
    for u in range(source, n):
        du = dist[u]
        if du is INF or du == INF:
            continue
        for v, w in adj[u]:
            nd = du + w
            if nd < dist[v]:
                dist[v] = nd
                pred[v] = u
    return dist, pred
