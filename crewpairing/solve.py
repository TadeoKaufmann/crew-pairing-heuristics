"""Column generation, the integer phase, and two reference methods."""

import time
from dataclasses import dataclass, field

import numpy as np

from .master import solve_integer, solve_lp
from .pricing import price


@dataclass
class Solution:
    lp_bound: float  # value of the final restricted LP
    objective: float  # integer solution
    pairings: list = field(default_factory=list)  # chosen pairings
    iterations: int = 0
    columns: int = 0
    cg_seconds: float = 0.0
    mip_seconds: float = 0.0
    mip_proven: bool | None = None  # False: time limit hit. None: no integer solve.

    @property
    def gap(self) -> float:
        """(objective - lp_bound) / objective. A true optimality gap only if pricing is exact."""
        return (self.objective - self.lp_bound) / self.objective


def column_generation(net, max_legs=4, pricing="exact", columns_per_iteration=1, max_iterations=100_000):
    """Grow the restricted master from one single-flight pairing per flight until pricing
    finds nothing with negative reduced cost. Returns (lp_bound, pairings, costs, iterations)."""
    pairings = [(f,) for f in range(net.n)]
    costs = [net.cost(p) for p in pairings]
    known = set(pairings)
    for iteration in range(1, max_iterations + 1):
        bound, _, duals = solve_lp(net.n, pairings, costs)
        proposed = price(net, duals, max_legs, pricing, columns_per_iteration)
        new = [p for _, p in proposed if p not in known]
        if not new:
            return bound, pairings, costs, iteration
        for p in new:
            pairings.append(p)
            known.add(p)
            costs.append(net.cost(p))
    raise RuntimeError("column generation did not converge")


def solve(net, max_legs=4, pricing="exact", columns_per_iteration=1, mip_time_limit=60.0) -> Solution:
    """Column generation followed by a binary solve over the generated columns
    ("price and branch": no new columns are priced during the integer phase)."""
    start = time.perf_counter()
    bound, pairings, costs, iterations = column_generation(net, max_legs, pricing, columns_per_iteration)
    cg_seconds = time.perf_counter() - start
    objective, chosen, proven = solve_integer(net.n, pairings, costs, mip_time_limit)
    return Solution(
        lp_bound=bound,
        objective=objective,
        pairings=[pairings[i] for i in chosen],
        iterations=iterations,
        columns=len(pairings) - net.n,
        cg_seconds=cg_seconds,
        mip_seconds=time.perf_counter() - start - cg_seconds,
        mip_proven=proven,
    )


def greedy(net, max_legs=4) -> Solution:
    """Reference heuristic: take flights in departure order and keep appending the
    cheapest free connection, then cut the chain at its cheapest prefix per flight."""
    free = np.ones(net.n, dtype=bool)
    chosen = []
    for first in range(net.n):
        if not free[first]:
            continue
        chain = [first]
        while len(chain) < max_legs:
            options = [int(j) for j in net.successors(chain[-1]) if free[j]]
            if not options:
                break
            chain.append(min(options, key=lambda j: net.cost(chain + [j])))
        length = min(range(1, len(chain) + 1), key=lambda k: net.cost(chain[:k]) / k)
        free[chain[:length]] = False
        chosen.append(tuple(chain[:length]))
    objective = sum(net.cost(p) for p in chosen)
    return Solution(lp_bound=float("nan"), objective=objective, pairings=chosen)


def enumerate_pairings(net, max_legs=4):
    """Every feasible pairing. Exponential: only for small instances and for testing."""
    successors = [net.successors(i) for i in range(net.n)]
    out, frontier = [], [(f,) for f in range(net.n)]
    for _ in range(max_legs):
        out.extend(frontier)
        frontier = [p + (int(j),) for p in frontier for j in successors[p[-1]]]
    return out
