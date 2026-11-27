"""The set partitioning master problem, solved with HiGHS through SciPy.

    min  sum_p c_p y_p    s.t.  every flight covered exactly once,  y >= 0 (or binary)
"""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csc_matrix


def coverage_matrix(n_flights, pairings) -> csc_matrix:
    """A[f, p] = 1 when pairing p contains flight f."""
    rows = np.fromiter((f for p in pairings for f in p), dtype=np.int64)
    indptr = np.concatenate([[0], np.cumsum([len(p) for p in pairings])])
    return csc_matrix((np.ones(len(rows)), rows, indptr), shape=(n_flights, len(pairings)))


def solve_lp(n_flights, pairings, costs):
    """LP relaxation. Returns (objective, y, duals), one dual per flight."""
    res = linprog(
        costs,
        A_eq=coverage_matrix(n_flights, pairings),
        b_eq=np.ones(n_flights),
        bounds=(0, None),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"master LP failed: {res.message}")
    return float(res.fun), res.x, np.asarray(res.eqlin.marginals)


def solve_integer(n_flights, pairings, costs, time_limit=60.0):
    """Binary master restricted to the given pairings.

    Returns (objective, chosen indices, proven). `proven` is False when HiGHS hit the
    time limit and the solution is only the best one found by then.
    """
    res = milp(
        costs,
        constraints=LinearConstraint(coverage_matrix(n_flights, pairings), 1, 1),
        integrality=np.ones(len(pairings)),
        bounds=Bounds(0, 1),
        options={"time_limit": time_limit},
    )
    if res.x is None:
        raise RuntimeError(f"no integer solution found: {res.message}")
    return float(res.fun), np.flatnonzero(res.x > 0.5), res.status == 0
