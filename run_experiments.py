"""Compare pricing strategies on synthetic schedules, or on your own instance folder.

    python run_experiments.py                                   # synthetic, writes results/synthetic.csv
    python run_experiments.py --instance path/to/instance1      # writes results/instance1.csv
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from crewpairing import CostModel, Network, greedy, load_instance, solve, synthetic_instance

SIZES = [(100, 4), (100, 5), (200, 4), (200, 5), (250, 4), (250, 5)]
PENALTY = 10_000.0


def describe(net, result, seconds):
    closed = sum(net.is_closed(p) for p in result.pairings)
    return {
        "objective": result.objective,
        "lp_bound": result.lp_bound,
        "gap_pct": 100 * result.gap,
        "pairings": len(result.pairings),
        "closed": closed,
        "avg_legs": net.n / len(result.pairings),
        "cg_iterations": result.iterations,
        "columns": result.columns,
        "integer_proven": result.mip_proven,
        "cg_seconds": result.cg_seconds,
        "seconds": seconds,
    }


def run(flights, max_legs, label):
    rows = []
    for penalty in (0.0, PENALTY):
        net = Network(flights, CostModel(open_penalty=penalty))
        singles = sum(net.cost((f,)) for f in range(net.n))
        # (method name, pricing, columns per iteration)
        methods = [("cg_exact", "exact", 1), ("cg_exact_multi", "exact", 100)]
        if penalty:
            methods.append(("cg_ranked", "ranked", 1))
        runs = []
        t = time.perf_counter()
        base = greedy(net, max_legs)
        runs.append(("greedy", base, time.perf_counter() - t))
        for name, pricing, k in methods:
            t = time.perf_counter()
            result = solve(net, max_legs, pricing, columns_per_iteration=k)
            runs.append((name, result, time.perf_counter() - t))
        for name, result, seconds in runs:
            row = {"instance": label, "flights": net.n, "max_legs": max_legs, "open_penalty": penalty, "method": name}
            row.update(describe(net, result, seconds))
            row["saving_vs_single_legs_pct"] = 100 * (1 - result.objective / singles)
            rows.append(row)
            print(
                f"{label} F={net.n} L={max_legs} M={penalty:>7.0f} {name:<15} "
                f"obj={row['objective']:>12,.1f} lb={row['lp_bound']:>12,.1f} "
                f"it={row['cg_iterations']:>5} t={seconds:7.2f}s"
                + ("  (integer time limit)" if result.mip_proven is False else ""),
                flush=True,
            )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instance", help="folder with day_*.csv files (not included in this repository)")
    p.add_argument("--max-days", type=int, default=10)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    label = Path(args.instance).name if args.instance else "synthetic"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{label}.csv"

    rows = []
    for n, max_legs in SIZES:
        if args.instance:
            flights = load_instance(args.instance, max_flights=n, max_days=args.max_days)
        else:
            flights = synthetic_instance(n, n_spokes=6 + n // 50, n_days=4, seed=n)
        rows += run(flights, max_legs, label)
        # saved after every size: the largest ones take a while
        pd.DataFrame(rows).to_csv(path, index=False, float_format="%.4f")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
