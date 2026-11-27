import numpy as np
import pandas as pd
import pytest

from crewpairing import (
    CostModel,
    Network,
    column_generation,
    enumerate_pairings,
    greedy,
    load_instance,
    solve,
    synthetic_instance,
    write_instance,
)
from crewpairing.master import solve_integer, solve_lp
from crewpairing.pricing import price

PENALTY = CostModel(open_penalty=10_000)


def schedule(*legs):
    """legs: (origin, destination, 'HH:MM', 'HH:MM') on the same day."""
    rows = [
        (f"F{i}", o, d, pd.Timestamp(f"2000-01-01 {a}"), pd.Timestamp(f"2000-01-01 {b}"))
        for i, (o, d, a, b) in enumerate(legs)
    ]
    return pd.DataFrame(rows, columns=["flight_id", "origin", "destination", "departure", "arrival"])


def test_connections_respect_airport_and_turnaround():
    net = Network(
        schedule(
            ("A", "B", "08:00", "09:00"),
            ("B", "A", "09:20", "10:20"),  # only 20 minutes: too tight
            ("B", "A", "09:30", "10:30"),  # exactly the minimum
            ("C", "A", "11:00", "12:00"),  # wrong airport
        )
    )
    assert not net.has_arc(0, 1)
    assert net.has_arc(0, 2)
    assert not net.has_arc(0, 3)


def test_cost_of_a_pairing():
    net = Network(schedule(("A", "B", "08:00", "09:00"), ("B", "A", "10:00", "11:30")), PENALTY)
    # 400 start + 150 flying + 20 connection + 0.1 * (60 - 30) waiting
    assert net.cost((0, 1)) == pytest.approx(573.0)
    # a single open leg pays the penalty
    assert net.cost((0,)) == pytest.approx(10_460.0)


def test_infeasible_pairing_is_rejected():
    net = Network(schedule(("A", "B", "08:00", "09:00"), ("C", "A", "10:00", "11:00")))
    with pytest.raises(KeyError):
        net.cost((0, 1))


def brute_force_best_reduced_cost(net, duals, max_legs):
    return min(net.cost(p) - duals[list(p)].sum() for p in enumerate_pairings(net, max_legs))


@pytest.mark.parametrize("penalty", [0.0, 10_000.0])
@pytest.mark.parametrize("seed", range(6))
def test_exact_pricing_matches_brute_force(seed, penalty):
    net = Network(synthetic_instance(40, seed=seed), CostModel(open_penalty=penalty))
    rng = np.random.default_rng(seed)
    duals = rng.uniform(0, 900 + penalty * 0.6, net.n)
    truth = brute_force_best_reduced_cost(net, duals, 4)
    found = price(net, duals, 4, "exact")
    if truth < -1e-6:
        assert found[0][0] == pytest.approx(truth)
        pairing = found[0][1]
        assert net.cost(pairing) - duals[list(pairing)].sum() == pytest.approx(truth)
    else:
        assert found == []


def test_ranked_pricing_can_miss_the_best_closed_pairing():
    # Two ways to reach the last flight with two legs: the cheap one starts away from
    # its destination (open), the dearer one closes the loop.
    net = Network(
        schedule(
            ("A", "B", "06:00", "07:00"),  # closes with flight 2, long wait
            ("C", "B", "08:00", "09:00"),  # short wait, but open
            ("B", "A", "10:00", "11:00"),
        ),
        PENALTY,
    )
    duals = np.array([300.0, 300.0, 400.0])
    assert price(net, duals, 2, "ranked") == []
    (value, pairing), = price(net, duals, 2, "exact")
    assert pairing == (0, 2) and value < 0


@pytest.mark.parametrize("penalty", [0.0, 10_000.0])
def test_column_generation_reaches_the_full_lp_optimum(penalty):
    net = Network(synthetic_instance(45, seed=3), CostModel(open_penalty=penalty))
    everything = enumerate_pairings(net, 4)
    full_lp, _, _ = solve_lp(net.n, everything, [net.cost(p) for p in everything])
    bound, pairings, costs, _ = column_generation(net, 4, "exact")
    assert bound == pytest.approx(full_lp, rel=1e-8)
    assert len(pairings) < len(everything)


def test_several_columns_per_iteration_same_bound_fewer_iterations():
    net = Network(synthetic_instance(60, seed=1))
    one = column_generation(net, 4, columns_per_iteration=1)
    many = column_generation(net, 4, columns_per_iteration=50)
    assert many[0] == pytest.approx(one[0], rel=1e-8)
    assert many[3] < one[3]


@pytest.mark.parametrize("seed", range(3))
def test_solution_covers_every_flight_once_and_is_bounded(seed):
    net = Network(synthetic_instance(60, seed=seed), PENALTY)
    result = solve(net, 4)
    covered = sorted(f for p in result.pairings for f in p)
    assert covered == list(range(net.n))
    assert result.objective == pytest.approx(sum(net.cost(p) for p in result.pairings))
    everything = enumerate_pairings(net, 4)
    optimum, _, proven = solve_integer(net.n, everything, [net.cost(p) for p in everything])
    assert proven and result.mip_proven
    assert result.lp_bound <= optimum + 1e-6 <= result.objective + 2e-6


def test_greedy_is_feasible_and_no_better_than_the_bound():
    net = Network(synthetic_instance(80, seed=2))
    baseline = greedy(net, 4)
    assert sorted(f for p in baseline.pairings for f in p) == list(range(net.n))
    assert max(len(p) for p in baseline.pairings) <= 4
    assert baseline.objective >= column_generation(net, 4)[0] - 1e-6


def test_synthetic_instances_are_reproducible_and_round_trip(tmp_path):
    a, b = synthetic_instance(50, seed=7), synthetic_instance(50, seed=7)
    pd.testing.assert_frame_equal(a, b)
    write_instance(a, tmp_path)
    loaded = load_instance(tmp_path)
    assert len(loaded) == 50
    assert Network(loaded).cost((0,)) == pytest.approx(Network(a).cost((0,)))
    assert len(load_instance(tmp_path, max_flights=10)) == 10
