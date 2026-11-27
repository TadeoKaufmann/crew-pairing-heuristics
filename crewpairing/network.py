"""The flight connection network and the pairing cost model."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    start: float = 400.0  # fixed cost of opening a pairing
    per_minute: float = 1.0  # per minute of flying
    per_connection: float = 20.0  # each additional leg
    per_wait_minute: float = 0.1  # waiting beyond the minimum turnaround
    turnaround: float = 30.0  # minimum minutes between consecutive flights
    open_penalty: float = 0.0  # added when a pairing does not end where it started


class Network:
    """Flights are nodes; an arc i -> j means j can follow i.

    Flights must be sorted by departure time, which makes index order a topological order.
    """

    def __init__(self, flights: pd.DataFrame, costs: CostModel = CostModel()):
        if not flights["departure"].is_monotonic_increasing:
            raise ValueError("flights must be sorted by departure")
        if (flights["arrival"] <= flights["departure"]).any():
            raise ValueError("every flight must arrive after it departs")
        self.flights = flights.reset_index(drop=True)
        self.costs = costs
        self.n = len(flights)

        airports, codes = np.unique(
            np.concatenate([flights["origin"].to_numpy(str), flights["destination"].to_numpy(str)]),
            return_inverse=True,
        )
        self.airports = airports
        self.origin, self.destination = codes[: self.n], codes[self.n :]

        epoch = flights["departure"].iloc[0]
        dep = ((flights["departure"] - epoch).dt.total_seconds() / 60).to_numpy()
        arr = ((flights["arrival"] - epoch).dt.total_seconds() / 60).to_numpy()
        self.node_cost = costs.per_minute * (arr - dep)

        wait = dep[None, :] - arr[:, None]  # wait[i, j]
        feasible = (self.destination[:, None] == self.origin[None, :]) & (wait >= costs.turnaround)
        self.tails, self.heads = np.nonzero(feasible)
        self.arc_cost = costs.per_connection + costs.per_wait_minute * (
            wait[self.tails, self.heads] - costs.turnaround
        )
        self._arcs = {(int(i), int(j)): float(c) for i, j, c in zip(self.tails, self.heads, self.arc_cost)}

    def has_arc(self, i, j) -> bool:
        return (i, j) in self._arcs

    def successors(self, i):
        return self.heads[self.tails == i]

    def is_closed(self, pairing) -> bool:
        return bool(self.origin[pairing[0]] == self.destination[pairing[-1]])

    def cost(self, pairing) -> float:
        """Full cost of a feasible pairing (a sequence of flight indices)."""
        total = self.costs.start + float(self.node_cost[list(pairing)].sum())
        for i, j in zip(pairing, pairing[1:]):
            total += self._arcs[int(i), int(j)]  # KeyError if the connection is infeasible
        if not self.is_closed(pairing):
            total += self.costs.open_penalty
        return total
