"""Flight data: loading day_*.csv schedules and generating synthetic ones.

The loader reads the layout used by the Kasirzadeh et al. (GERAD) crew scheduling
instances, which are not redistributed here:

    # leg_nb , airport_dep , date_dep , hour_dep , airport_arr , date_arr , hour_arr
"""

import glob
import os

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "leg_nb",
    "airport_dep",
    "date_dep",
    "hour_dep",
    "airport_arr",
    "date_arr",
    "hour_arr",
]


def load_instance(folder, max_flights=None, max_days=None) -> pd.DataFrame:
    """Read day_1.csv, day_2.csv, ... and return flights sorted by departure.

    Keeps the first `max_days` files and then the first `max_flights` departures.
    """
    files = sorted(
        glob.glob(os.path.join(folder, "day_*.csv")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]),
    )
    if not files:
        raise FileNotFoundError(f"no day_*.csv files in {folder}")
    frames = [
        pd.read_csv(f, engine="python", sep=r"\s*,\s*", header=None, names=RAW_COLUMNS, comment="#")
        for f in files[:max_days]
    ]
    raw = pd.concat(frames, ignore_index=True).dropna(how="all")
    flights = pd.DataFrame(
        {
            "flight_id": raw["leg_nb"],
            "origin": raw["airport_dep"],
            "destination": raw["airport_arr"],
            "departure": pd.to_datetime(raw["date_dep"].str.strip() + " " + raw["hour_dep"].str.strip()),
            "arrival": pd.to_datetime(raw["date_arr"].str.strip() + " " + raw["hour_arr"].str.strip()),
        }
    )
    flights = flights.sort_values("departure").reset_index(drop=True)
    return flights.iloc[:max_flights] if max_flights else flights


def synthetic_instance(n_flights=100, n_spokes=6, n_hubs=2, n_days=4, seed=0) -> pd.DataFrame:
    """A made-up hub-and-spoke schedule.

    Most flights come in rotations that leave a hub and return to it (out-and-back or
    triangles), so closed pairings exist. About one flight in eight is a one-way leg,
    so some pairings cannot be closed. This is NOT airline data.
    """
    if n_flights < 1:
        raise ValueError("n_flights must be positive")
    rng = np.random.default_rng(seed)
    hubs = [f"HUB{i + 1}" for i in range(n_hubs)]
    spokes = [f"SP{i + 1:02d}" for i in range(n_spokes)]
    airports = hubs + spokes
    # one flight time per airport pair, in minutes
    duration = {}
    for i, a in enumerate(airports):
        for b in airports[i + 1 :]:
            duration[a, b] = duration[b, a] = int(rng.integers(45, 181))

    start = pd.Timestamp("2000-01-01")
    legs = []
    while len(legs) < n_flights:
        day = start + pd.Timedelta(days=int(rng.integers(n_days)))
        clock = day + pd.Timedelta(minutes=int(rng.integers(6 * 60, 15 * 60)))
        kind = rng.random()
        hub = str(rng.choice(hubs))
        if kind < 0.1:
            route = [str(a) for a in rng.choice(airports, 2, replace=False)]
        elif kind < 0.7:
            route = [hub, str(rng.choice(spokes)), hub]
        else:
            a, b = (str(s) for s in rng.choice(spokes, 2, replace=False))
            route = [hub, a, b, hub]
        for origin, destination in zip(route, route[1:]):
            arrival = clock + pd.Timedelta(minutes=duration[origin, destination])
            legs.append((origin, destination, clock, arrival))
            clock = arrival + pd.Timedelta(minutes=int(rng.integers(35, 151)))

    flights = pd.DataFrame(legs[:n_flights], columns=["origin", "destination", "departure", "arrival"])
    flights = flights.sort_values(["departure", "arrival", "origin"]).reset_index(drop=True)
    flights.insert(0, "flight_id", [f"F{i:04d}" for i in range(len(flights))])
    return flights


def write_instance(flights: pd.DataFrame, folder) -> None:
    """Write a schedule as one day_<n>.csv file per departure day."""
    os.makedirs(folder, exist_ok=True)
    days = flights["departure"].dt.normalize()
    for n, day in enumerate(sorted(days.unique()), start=1):
        part = flights[days == day]
        with open(os.path.join(folder, f"day_{n}.csv"), "w", newline="\n") as f:
            f.write("# " + " , ".join(RAW_COLUMNS) + "\n")
            for r in part.itertuples():
                f.write(
                    f"{r.flight_id} , {r.origin} , {r.departure:%Y-%m-%d} , {r.departure:%H:%M} , "
                    f"{r.destination} , {r.arrival:%Y-%m-%d} , {r.arrival:%H:%M}\n"
                )
