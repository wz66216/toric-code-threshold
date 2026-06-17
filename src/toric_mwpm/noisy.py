from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from math import sqrt
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
from networkx.algorithms.matching import min_weight_matching

from .decoder import decode_mwpm
from .lattice import ToricChain, random_error, shortest_path_chain


@dataclass(frozen=True, order=True)
class DetectionEvent:
    x: int
    y: int
    t: int


def _likelihood_weight(probability: float) -> float:
    if not 0 <= probability < 0.5:
        raise ValueError("probability must satisfy 0 <= p < 0.5")
    if probability == 0:
        return math.inf
    return math.log((1.0 - probability) / probability)


def _validate_square_layer(layer: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(layer, dtype=bool)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] <= 0:
        raise ValueError(f"{name} must be a nonempty square 2D array")
    return array


def measured_syndrome_history(
    true_syndromes: list[np.ndarray] | tuple[np.ndarray, ...],
    measurement_flips: list[np.ndarray] | tuple[np.ndarray, ...],
) -> list[np.ndarray]:
    if len(true_syndromes) < 1:
        raise ValueError("true_syndromes must include at least one layer")
    if len(measurement_flips) != len(true_syndromes) - 1:
        raise ValueError("measurement_flips must have one layer per noisy round")

    initial = _validate_square_layer(true_syndromes[0], name="true_syndromes[0]")
    L = initial.shape[0]
    measured = [initial.copy()]
    for i, (syndrome, flips) in enumerate(zip(true_syndromes[1:], measurement_flips)):
        s = _validate_square_layer(syndrome, name=f"true_syndromes[{i + 1}]")
        f = _validate_square_layer(flips, name=f"measurement_flips[{i}]")
        if s.shape != (L, L) or f.shape != (L, L):
            raise ValueError("all layers must have the same shape")
        measured.append(np.logical_xor(s, f))
    return measured


def detection_events_from_measurements(measured_syndromes) -> list[DetectionEvent]:
    if len(measured_syndromes) < 2:
        return []
    previous = _validate_square_layer(measured_syndromes[0], name="measured_syndromes[0]")
    events: list[DetectionEvent] = []
    for t, current_layer in enumerate(measured_syndromes[1:]):
        current = _validate_square_layer(current_layer, name=f"measured_syndromes[{t + 1}]")
        if current.shape != previous.shape:
            raise ValueError("all measured syndrome layers must have the same shape")
        for x, y in np.argwhere(np.logical_xor(current, previous)):
            events.append(DetectionEvent(int(x), int(y), t))
        previous = current
    return sorted(events)


def spacetime_distance(
    a: DetectionEvent,
    b: DetectionEvent,
    *,
    L: int,
    space_weight: float,
    time_weight: float,
) -> float:
    dx_raw = abs(a.x - b.x)
    dy_raw = abs(a.y - b.y)
    dx = min(dx_raw, L - dx_raw)
    dy = min(dy_raw, L - dy_raw)
    spatial_steps = dx + dy
    temporal_steps = abs(a.t - b.t)

    space_cost = 0.0 if spatial_steps == 0 else space_weight * spatial_steps
    time_cost = 0.0 if temporal_steps == 0 else time_weight * temporal_steps

    if (spatial_steps > 0 and not math.isfinite(space_cost)) or (temporal_steps > 0 and not math.isfinite(time_cost)):
        return math.inf
    return float(space_cost + time_cost)


def _project_pair_to_recovery(a: DetectionEvent, b: DetectionEvent, L: int) -> ToricChain:
    if a.t <= b.t:
        earlier, later = a, b
    else:
        earlier, later = b, a
    return shortest_path_chain(L, (earlier.x, earlier.y), (later.x, later.y))


def match_spacetime_events(events, *, L: int, p: float, q: float) -> list[tuple[DetectionEvent, DetectionEvent]]:
    sorted_events = sorted(events)
    if len(sorted_events) % 2 != 0:
        raise ValueError("spacetime event list must contain an even number of events")
    if not sorted_events:
        return []

    space_weight = _likelihood_weight(p)
    time_weight = _likelihood_weight(q)

    graph = nx.Graph()
    for i, event in enumerate(sorted_events):
        graph.add_node(i, event=event)
    for i in range(len(sorted_events)):
        for j in range(i + 1, len(sorted_events)):
            weight = spacetime_distance(sorted_events[i], sorted_events[j], L=L, space_weight=space_weight, time_weight=time_weight)
            if math.isfinite(weight):
                graph.add_edge(i, j, weight=weight)

    matching = min_weight_matching(graph, weight="weight")
    if len(matching) * 2 != len(sorted_events):
        raise ValueError("no finite perfect matching for the given p/q constraints")
    return [(sorted_events[i], sorted_events[j]) for i, j in sorted(tuple(sorted(pair)) for pair in matching)]


def decode_spacetime_mwpm(events, *, L: int, p: float, q: float) -> ToricChain:
    recovery = ToricChain.empty(L)
    for a, b in match_spacetime_events(events, L=L, p=p, q=q):
        recovery = recovery + _project_pair_to_recovery(a, b, L)
    return recovery


@dataclass(frozen=True)
class NoisySimulationResult:
    L: int
    T: int
    p: float
    q: float
    trials: int
    failures: int

    @property
    def failure_rate(self) -> float:
        return self.failures / self.trials if self.trials else float("nan")

    @property
    def standard_error(self) -> float:
        if self.trials == 0:
            return float("nan")
        rate = self.failure_rate
        return sqrt(rate * (1.0 - rate) / self.trials)

    @property
    def wilson_interval_95(self) -> tuple[float, float]:
        if self.trials == 0:
            return float("nan"), float("nan")
        z = 1.959963984540054
        n = self.trials
        phat = self.failure_rate
        denom = 1.0 + (z * z) / n
        center = (phat + (z * z) / (2.0 * n)) / denom
        margin = (z / denom) * sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))
        low = max(0.0, center - margin)
        high = min(1.0, center + margin)
        return low, high


def _validate_run_parameters(L: int, T: int, p: float, q: float, trials: int) -> None:
    if L <= 0:
        raise ValueError("L must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if trials <= 0:
        raise ValueError("trials must be positive")
    _likelihood_weight(p)
    _likelihood_weight(q)


def _sample_noisy_history(*, L: int, T: int, p: float, q: float, rng: np.random.Generator):
    cumulative = ToricChain.empty(L)
    true_syndromes = [np.zeros((L, L), dtype=bool)]
    measurement_flips = []
    for _ in range(T):
        cumulative = cumulative + random_error(L, p, rng)
        true_syndromes.append(cumulative.syndrome())
        measurement_flips.append(rng.random((L, L)) < q)
    measured = measured_syndrome_history(true_syndromes, measurement_flips)
    measured.append(cumulative.syndrome())
    events = detection_events_from_measurements(measured)
    return cumulative, events


def sample_noisy_trial(*, L: int, T: int, p: float, q: float, seed: int | None = None):
    _validate_run_parameters(L, T, p, q, trials=1)
    rng = np.random.default_rng(seed)
    return _sample_noisy_history(L=L, T=T, p=p, q=q, rng=rng)


def run_noisy_point(
    *,
    L: int,
    T: int | None = None,
    p: float,
    q: float | None = None,
    trials: int,
    seed: int | None = None,
    decoder_p: float | None = None,
    decoder_q: float | None = None,
) -> NoisySimulationResult:
    T = L if T is None else T
    q = p if q is None else q
    _validate_run_parameters(L, T, p, q, trials)
    decoder_p = p if decoder_p is None else decoder_p
    decoder_q = q if decoder_q is None else decoder_q
    _likelihood_weight(decoder_p)
    _likelihood_weight(decoder_q)
    rng = np.random.default_rng(seed)
    failures = 0
    for _ in range(trials):
        total_error, events = _sample_noisy_history(L=L, T=T, p=p, q=q, rng=rng)
        recovery = decode_spacetime_mwpm(events, L=L, p=decoder_p, q=decoder_q)
        if (total_error + recovery).is_logical_failure():
            failures += 1
    return NoisySimulationResult(L=L, T=T, p=p, q=q, trials=trials, failures=failures)


def run_noisy_sweep(*, L_values: Iterable[int], p_values: Iterable[float], trials: int, q: float | None = None, T: int | None = None, seed: int | None = None) -> list[NoisySimulationResult]:
    L_values = list(L_values)
    p_values = list(p_values)
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(len(L_values) * len(p_values))
    results = []
    index = 0
    for L in L_values:
        for p in p_values:
            results.append(run_noisy_point(L=L, T=L if T is None else T, p=p, q=p if q is None else q, trials=trials, seed=int(child_seeds[index].generate_state(1)[0])))
            index += 1
    return results


def write_noisy_csv(results, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["L", "T", "p", "q", "trials", "failures", "failure_rate", "standard_error", "ci95_low", "ci95_high"])
        writer.writeheader()
        for result in results:
            ci95_low, ci95_high = result.wilson_interval_95
            writer.writerow({
                "L": result.L,
                "T": result.T,
                "p": result.p,
                "q": result.q,
                "trials": result.trials,
                "failures": result.failures,
                "failure_rate": result.failure_rate,
                "standard_error": result.standard_error,
                "ci95_low": ci95_low,
                "ci95_high": ci95_high,
            })


def plot_noisy_results(results, path: str | Path) -> None:
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    try:
        grouped = {}
        for result in results:
            grouped.setdefault((result.L, result.T), []).append(result)
        for (L, T) in sorted(grouped):
            subset = sorted(grouped[(L, T)], key=lambda r: r.p)
            lowers = []
            uppers = []
            for result in subset:
                low, high = result.wilson_interval_95
                rate = result.failure_rate
                lowers.append(max(0.0, rate - low))
                uppers.append(max(0.0, high - rate))
            ax.errorbar([r.p for r in subset], [r.failure_rate for r in subset], yerr=[lowers, uppers], marker="o", capsize=3, label=f"L={L}, T={T}")
        ax.set_xlabel("p")
        ax.set_ylabel("failure rate")
        if grouped:
            ax.legend()
        fig.tight_layout()
        fig.savefig(output)
    finally:
        plt.close(fig)
