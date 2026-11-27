"""README figures, computed from synthetic schedules (takes a few minutes).

    python make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from crewpairing import CostModel, Network, solve, synthetic_instance
from crewpairing.master import solve_lp
from crewpairing.pricing import price

FIGURES = Path("figures")
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
INK, MUTED, GRID = "#191919", "#666666", "#e6e6e6"
PENALISED = CostModel(open_penalty=10_000)

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
    }
)


def lp_history(net, max_legs, method, columns_per_iteration):
    """Value of the restricted LP at every column generation iteration."""
    pairings = [(f,) for f in range(net.n)]
    costs = [net.cost(p) for p in pairings]
    known = set(pairings)
    values = []
    while True:
        value, _, duals = solve_lp(net.n, pairings, costs)
        values.append(value)
        new = [p for _, p in price(net, duals, max_legs, method, columns_per_iteration) if p not in known]
        if not new:
            return values
        for p in new:
            pairings.append(p)
            known.add(p)
            costs.append(net.cost(p))


def convergence():
    net = Network(synthetic_instance(200, n_spokes=10, n_days=4, seed=200), PENALISED)
    runs = [
        ("exact pricing, 1 column per iteration", "exact", 1, BLUE),
        ("exact pricing, up to 100 columns per iteration", "exact", 100, AQUA),
        ("ranked pricing, 1 column per iteration", "ranked", 1, ORANGE),
    ]
    histories = [(label, lp_history(net, 4, method, k), colour) for label, method, k, colour in runs]
    optimum = histories[0][1][-1]

    floor = 1e-5  # percent; anything below counts as converged
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for label, values, colour in histories:
        excess = [max(100 * (v / optimum - 1), floor) for v in values]
        ax.plot(range(1, len(values) + 1), excess, color=colour, linewidth=2, label=label)
        ax.plot(len(values), excess[-1], "o", color=colour, markersize=6, markeredgecolor="white")
    stuck = 100 * (histories[2][1][-1] / optimum - 1)
    ax.annotate(
        f"stops {stuck:.2f}% above\nthe LP optimum",
        (len(histories[2][1]), stuck),
        xytext=(10, 22),
        textcoords="offset points",
        color=INK,
        arrowprops={"arrowstyle": "-", "color": MUTED},
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Column generation iteration")
    ax.set_ylabel("Restricted LP above optimum (%)")
    ax.grid(axis="y", color=GRID)
    ax.set_title("Synthetic schedule, 200 flights, up to 4 legs, open pairings penalised", pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
    fig.savefig(FIGURES / "convergence.png")
    plt.close(fig)


def pairings_timeline():
    flights = synthetic_instance(40, n_spokes=6, n_days=2, seed=4)
    net = Network(flights, PENALISED)
    result = solve(net, 4)
    pairings = sorted(result.pairings, key=lambda p: p[0])

    fig, ax = plt.subplots(figsize=(8, 0.28 * len(pairings) + 1.1))
    for row, pairing in enumerate(pairings):
        closed = net.is_closed(pairing)
        legs = flights.iloc[list(pairing)]
        ax.plot(
            [legs["departure"].iloc[0], legs["arrival"].iloc[-1]], [row, row], color=GRID, linewidth=1, zorder=1
        )
        for leg in legs.itertuples():
            start = mdates.date2num(leg.departure)
            width = mdates.date2num(leg.arrival) - start
            ax.barh(row, width, left=start, height=0.62, color=BLUE if closed else ORANGE, zorder=2)
    routes = [
        " ".join([flights["origin"][p[0]]] + [flights["destination"][f] for f in p]).replace(" ", " > ")
        for p in pairings
    ]
    ax.set_yticks(range(len(pairings)), routes, fontsize=7)
    ax.invert_yaxis()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("day %d\n%H:%M"))
    ax.tick_params(axis="y", length=0)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=BLUE, label="closed (ends where it started)"),
            plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="open (penalised)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=2,
    )
    ax.set_title(f"A solution for 40 synthetic flights: {len(pairings)} pairings, one bar per flight", pad=10)
    fig.savefig(FIGURES / "pairings.png")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    pairings_timeline()
    convergence()
    print(f"figures written to {FIGURES}")
