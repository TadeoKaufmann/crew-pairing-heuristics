"""Pricing: find pairings with negative reduced cost  c(p) - sum of duals over p.

Both strategies run the same dynamic program over states (last flight, number of legs).
They differ in how the penalty for open pairings is handled.

* "ranked" (the project's penalised variant): one DP ignores the penalty, keeps only the
  cheapest path per state, and adds the penalty afterwards. A closed path can be
  discarded because a cheaper open path reaches the same state, so improving columns
  can be missed.
* "exact": one DP per starting airport. All paths in it share the origin, so whether
  the pairing is closed depends only on the state, and nothing relevant is discarded.

With open_penalty = 0 both are exact.
"""

import numpy as np

TOL = 1e-6


def _dynamic_program(net, duals, max_legs, allowed_start):
    """dist[k, j]: cheapest reduced cost (without open penalty) of a path with k+1 legs
    ending at flight j. parent[k, j]: the previous flight on that path."""
    reduced = net.node_cost - duals
    dist = np.full((max_legs, net.n), np.inf)
    parent = np.full((max_legs, net.n), -1, dtype=np.int64)
    dist[0, allowed_start] = net.costs.start + reduced[allowed_start]
    step = reduced[net.heads] + net.arc_cost
    for k in range(1, max_legs):
        candidate = dist[k - 1, net.tails] + step
        usable = np.flatnonzero(np.isfinite(candidate))
        if len(usable) == 0:
            break
        # for every head flight, the usable arc with the smallest candidate value
        order = usable[np.lexsort((candidate[usable], net.heads[usable]))]
        first = np.ones(len(order), dtype=bool)
        first[1:] = net.heads[order][1:] != net.heads[order][:-1]
        best = order[first]
        dist[k, net.heads[best]] = candidate[best]
        parent[k, net.heads[best]] = net.tails[best]
    return dist, parent


def _path(parent, k, j):
    path = [int(j)]
    while k > 0:
        j = parent[k, j]
        k -= 1
        path.append(int(j))
    return tuple(reversed(path))


def price(net, duals, max_legs=4, method="exact", max_columns=1):
    """Return up to `max_columns` pairs (reduced_cost, pairing), best first.

    Only pairings with negative reduced cost are returned. With max_columns > 1, at most
    one pairing per final flight is proposed, which keeps the new columns diverse.
    """
    if method not in ("exact", "ranked"):
        raise ValueError(f"unknown pricing method: {method}")
    duals = np.asarray(duals, dtype=float)
    penalty = net.costs.open_penalty
    if method == "exact" and penalty > 0:
        groups = [np.flatnonzero(net.origin == a) for a in np.unique(net.origin)]
    else:
        groups = [np.arange(net.n)]

    best = {}  # final flight -> (reduced cost, pairing)
    for starts in groups:
        dist, parent = _dynamic_program(net, duals, max_legs, starts)
        # the penalty can only increase a reduced cost, so filter on the base value first
        for k, j in zip(*np.nonzero(dist < -TOL)):
            value = float(dist[k, j])
            if j in best and value >= best[j][0]:
                continue
            if penalty > 0:
                if method == "exact":
                    start = net.origin[starts[0]]
                    value += penalty * float(net.destination[j] != start)
                else:
                    value += penalty * float(not net.is_closed(_path(parent, k, j)))
            if value < -TOL and (j not in best or value < best[j][0]):
                best[j] = (value, _path(parent, k, j))
    return sorted(best.values())[:max_columns]
