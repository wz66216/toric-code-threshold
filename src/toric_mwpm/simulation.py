from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .decoder import decode_mwpm
from .lattice import random_error


@dataclass(frozen=True)
class SimulationResult:
    L: int
    p: float
    trials: int
    failures: int

    @property
    def failure_rate(self) -> float:
        return self.failures / self.trials if self.trials > 0 else math.nan

    @property
    def standard_error(self) -> float:
        if self.trials <= 0:
            return math.nan
        rate = self.failure_rate
        return math.sqrt(rate * (1.0 - rate) / self.trials)


def run_point(L: int, p: float, trials: int, seed: int | None = None) -> SimulationResult:
    if trials <= 0:
        raise ValueError("trials must be positive")
    rng = np.random.default_rng(seed)
    failures = 0
    for _ in range(trials):
        error = random_error(L, p, rng)
        recovery = decode_mwpm(error.syndrome())
        if (error + recovery).is_logical_failure():
            failures += 1
    return SimulationResult(L=L, p=p, trials=trials, failures=failures)


def run_sweep(L_values, p_values, trials: int, seed: int | None = None):
    L_list = list(L_values)
    p_list = list(p_values)
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(len(L_list) * len(p_list))
    results = []
    idx = 0
    for L in L_list:
        for p in p_list:
            results.append(run_point(L, p, trials, seed=int(child_seeds[idx].generate_state(1)[0])))
            idx += 1
    return results


def write_csv(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["L", "p", "trials", "failures", "failure_rate", "standard_error"])
        for result in results:
            writer.writerow([result.L, result.p, result.trials, result.failures, result.failure_rate, result.standard_error])


def estimate_crossing(results):
    by_L = {}
    for result in results:
        by_L.setdefault(result.L, {})[result.p] = result.failure_rate
    if len(by_L) < 2:
        return None
    L_min = min(by_L)
    L_max = max(by_L)
    common = sorted(set(by_L[L_min]) & set(by_L[L_max]))
    if not common:
        return None
    best_p = min(common, key=lambda p: abs(by_L[L_min][p] - by_L[L_max][p]))
    return best_p


def plot_results(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    by_L = {}
    for result in results:
        by_L.setdefault(result.L, []).append(result)
    fig, ax = plt.subplots()
    try:
        for L, group in sorted(by_L.items()):
            group = sorted(group, key=lambda r: r.p)
            ax.errorbar([r.p for r in group], [r.failure_rate for r in group], yerr=[r.standard_error for r in group], marker="o", label=f"L={L}")
        crossing = estimate_crossing(results)
        if crossing is not None:
            ax.axvline(crossing, linestyle="--", color="black", alpha=0.6)
        ax.set_xlabel("Physical error rate p")
        ax.set_ylabel("Logical failure rate")
        ax.set_title("Toric MWPM simulation")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)
