# Noisy Syndrome 3D MWPM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a phenomenological noisy-syndrome simulator that decodes repeated toric-code syndrome measurements using a 3D spacetime MWPM graph.

**Architecture:** Keep the existing perfect-measurement 2D simulator intact. Add a new `toric_mwpm.noisy` module for spacetime events, noisy measurement histories, spacetime MWPM, Monte Carlo sweeps, CSV output, and plotting. Add a separate `scripts/run_noisy_threshold.py` CLI and README/report guidance so the 3D extension can be run independently of the 2D baseline.

**Tech Stack:** Python, NumPy, NetworkX MWPM, Matplotlib, pytest, existing `toric_mwpm.lattice` and `toric_mwpm.decoder` helpers.

---

## File Structure

- Create `src/toric_mwpm/noisy.py`: spacetime data structures, deterministic history helpers, noisy MWPM decoder, Monte Carlo simulation, CSV writer, plotter.
- Modify `src/toric_mwpm/__init__.py`: export the public noisy-syndrome APIs.
- Create `scripts/run_noisy_threshold.py`: command-line entrypoint for noisy-syndrome threshold sweeps.
- Create `tests/test_noisy.py`: deterministic unit tests for detection events, spacetime distances, projected recovery, simulations, CSV/plot behavior.
- Create `tests/test_run_noisy_threshold.py`: CLI smoke test.
- Modify `README.md`: document the noisy-syndrome extension, interpretation, and recommended small/report sweeps.

---

### Task 1: Spacetime Data Structures and Detection Events

**Files:**
- Create: `src/toric_mwpm/noisy.py`
- Modify: `src/toric_mwpm/__init__.py`
- Test: `tests/test_noisy.py`

- [ ] **Step 1: Write failing tests for deterministic detection-event logic**

Create `tests/test_noisy.py` with:

```python
import numpy as np

from toric_mwpm.lattice import ToricChain, chain_from_edges
from toric_mwpm.noisy import (
    DetectionEvent,
    detection_events_from_measurements,
    measured_syndrome_history,
)


def test_single_measurement_error_creates_adjacent_time_events():
    L = 3
    measurement_flips = [
        np.zeros((L, L), dtype=bool),
        np.zeros((L, L), dtype=bool),
    ]
    measurement_flips[0][1, 2] = True
    true_syndromes = [
        np.zeros((L, L), dtype=bool),
        np.zeros((L, L), dtype=bool),
        np.zeros((L, L), dtype=bool),
    ]

    measured = measured_syndrome_history(true_syndromes, measurement_flips)
    events = detection_events_from_measurements(measured)

    assert events == [DetectionEvent(1, 2, 0), DetectionEvent(1, 2, 1)]


def test_single_data_error_creates_spatial_detection_events():
    L = 4
    error = chain_from_edges(L, horizontal_edges=[(1, 2)])
    true_syndromes = [
        np.zeros((L, L), dtype=bool),
        error.syndrome(),
    ]
    measurement_flips = [np.zeros((L, L), dtype=bool)]

    measured = measured_syndrome_history(true_syndromes, measurement_flips)
    events = detection_events_from_measurements(measured)

    assert events == [DetectionEvent(1, 2, 0), DetectionEvent(2, 2, 0)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_noisy.py -v`

Expected: FAIL with `ModuleNotFoundError` or missing symbols from `toric_mwpm.noisy`.

- [ ] **Step 3: Implement deterministic history helpers**

Create `src/toric_mwpm/noisy.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from pathlib import Path
from typing import Iterable, Sequence
import csv

import networkx as nx
import numpy as np

from .lattice import ToricChain, chain_from_edges, random_error, shortest_path_chain


@dataclass(frozen=True, order=True)
class DetectionEvent:
    x: int
    y: int
    t: int


def _validate_square_layer(layer: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(layer, dtype=bool)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] <= 0:
        raise ValueError(f"{name} must be a nonempty square 2D array")
    return array


def measured_syndrome_history(
    true_syndromes: Sequence[np.ndarray],
    measurement_flips: Sequence[np.ndarray],
) -> list[np.ndarray]:
    if len(true_syndromes) < 1:
        raise ValueError("true_syndromes must include at least the initial all-zero layer")
    if len(measurement_flips) != len(true_syndromes) - 1:
        raise ValueError("measurement_flips must contain one layer per noisy measurement round")

    initial = _validate_square_layer(true_syndromes[0], name="true_syndromes[0]")
    measured: list[np.ndarray] = [initial.copy()]
    L = initial.shape[0]

    for index, (syndrome, flips) in enumerate(zip(true_syndromes[1:], measurement_flips)):
        s = _validate_square_layer(syndrome, name=f"true_syndromes[{index + 1}]")
        f = _validate_square_layer(flips, name=f"measurement_flips[{index}]")
        if s.shape != (L, L) or f.shape != (L, L):
            raise ValueError("all syndrome and flip layers must have the same shape")
        measured.append(np.logical_xor(s, f))

    return measured


def detection_events_from_measurements(measured_syndromes: Sequence[np.ndarray]) -> list[DetectionEvent]:
    if len(measured_syndromes) < 2:
        return []
    previous = _validate_square_layer(measured_syndromes[0], name="measured_syndromes[0]")
    events: list[DetectionEvent] = []
    for t, current_layer in enumerate(measured_syndromes[1:]):
        current = _validate_square_layer(current_layer, name=f"measured_syndromes[{t + 1}]")
        if current.shape != previous.shape:
            raise ValueError("all measured syndrome layers must have the same shape")
        changed = np.logical_xor(current, previous)
        for x, y in np.argwhere(changed):
            events.append(DetectionEvent(int(x), int(y), t))
        previous = current
    return sorted(events)
```

Modify `src/toric_mwpm/__init__.py` to export:

```python
from .noisy import DetectionEvent
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_noisy.py -v`

Expected: 2 passed.

- [ ] **Step 5: Run full regression suite**

Run: `python -m pytest -q`

Expected: all existing tests plus `tests/test_noisy.py` pass.

---

### Task 2: Spacetime Distance, Path Projection, and MWPM Decoder

**Files:**
- Modify: `src/toric_mwpm/noisy.py`
- Test: `tests/test_noisy.py`

- [ ] **Step 1: Add failing tests for distance, empty decode, and projection invariants**

Append to `tests/test_noisy.py`:

```python
from toric_mwpm.decoder import decode_mwpm
from toric_mwpm.noisy import (
    decode_spacetime_mwpm,
    spacetime_distance,
)


def test_spacetime_distance_is_periodic_in_space_and_open_in_time():
    a = DetectionEvent(0, 0, 0)
    b = DetectionEvent(4, 0, 0)
    c = DetectionEvent(0, 0, 3)

    assert spacetime_distance(a, b, L=5, space_weight=2.0, time_weight=7.0) == 2.0
    assert spacetime_distance(a, c, L=5, space_weight=2.0, time_weight=7.0) == 21.0


def test_empty_spacetime_decode_returns_empty_chain():
    recovery = decode_spacetime_mwpm([], L=4, p=0.03, q=0.03)
    assert recovery.weight() == 0
    assert recovery.logical_parity() == (False, False)


def test_q_zero_t_one_matches_existing_2d_decoder_for_fixed_error():
    L = 5
    error = chain_from_edges(L, horizontal_edges=[(1, 2), (4, 3)])
    measured = measured_syndrome_history(
        [np.zeros((L, L), dtype=bool), error.syndrome()],
        [np.zeros((L, L), dtype=bool)],
    )
    events = detection_events_from_measurements(measured)

    noisy_recovery = decode_spacetime_mwpm(events, L=L, p=0.08, q=0.0)
    baseline_recovery = decode_mwpm(error.syndrome())

    assert (error + noisy_recovery).syndrome().sum() == 0
    assert noisy_recovery.weight() == baseline_recovery.weight()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_noisy.py -v`

Expected: FAIL with missing `spacetime_distance` and `decode_spacetime_mwpm`.

- [ ] **Step 3: Implement spacetime distance, shortest path projection, and decoder**

Append to `src/toric_mwpm/noisy.py`:

```python
def _likelihood_weight(probability: float) -> float:
    if probability < 0 or probability >= 0.5:
        raise ValueError("probabilities must satisfy 0 <= p < 0.5 for MWPM weights")
    if probability == 0:
        return 1_000_000.0
    return log((1.0 - probability) / probability)


def spacetime_distance(
    a: DetectionEvent,
    b: DetectionEvent,
    *,
    L: int,
    space_weight: float,
    time_weight: float,
) -> float:
    if L <= 0:
        raise ValueError("L must be positive")
    dx = abs(a.x - b.x)
    dy = abs(a.y - b.y)
    spatial = min(dx, L - dx) + min(dy, L - dy)
    temporal = abs(a.t - b.t)
    return space_weight * spatial + time_weight * temporal


def _project_pair_to_recovery(a: DetectionEvent, b: DetectionEvent, L: int) -> ToricChain:
    if a.t <= b.t:
        start, end = a, b
    else:
        start, end = b, a
    recovery = ToricChain.empty(L)
    x, y = start.x, start.y
    # Time-like movement does not add data recovery edges. Spatial correction is
    # applied once at the later endpoint, matching the projected spacetime path.
    spatial_path = shortest_path_chain(L, (x, y), (end.x, end.y))
    return recovery + spatial_path


def decode_spacetime_mwpm(
    events: Sequence[DetectionEvent],
    *,
    L: int,
    p: float,
    q: float,
) -> ToricChain:
    if L <= 0:
        raise ValueError("L must be positive")
    events = sorted(events)
    if len(events) == 0:
        return ToricChain.empty(L)
    if len(events) % 2:
        raise ValueError("number of detection events must be even")

    ws = _likelihood_weight(p)
    wt = _likelihood_weight(q)
    graph = nx.Graph()
    for i, event in enumerate(events):
        graph.add_node(i, event=event)
    for i, a in enumerate(events):
        for j in range(i + 1, len(events)):
            b = events[j]
            graph.add_edge(
                i,
                j,
                weight=spacetime_distance(a, b, L=L, space_weight=ws, time_weight=wt),
            )

    matching = nx.algorithms.matching.min_weight_matching(graph, weight="weight")
    recovery = ToricChain.empty(L)
    for i, j in sorted(tuple(sorted(pair)) for pair in matching):
        recovery = recovery + _project_pair_to_recovery(events[i], events[j], L)
    return recovery
```

- [ ] **Step 4: Run targeted tests**

Run: `python -m pytest tests/test_noisy.py -v`

Expected: all noisy tests pass.

- [ ] **Step 5: Run full regression suite**

Run: `python -m pytest -q`

Expected: all tests pass.

---

### Task 3: Noisy Trial Generation and Monte Carlo Sweeps

**Files:**
- Modify: `src/toric_mwpm/noisy.py`
- Modify: `src/toric_mwpm/__init__.py`
- Test: `tests/test_noisy.py`

- [ ] **Step 1: Add failing tests for zero noise and reproducible noisy points**

Append to `tests/test_noisy.py`:

```python
from toric_mwpm.noisy import NoisySimulationResult, run_noisy_point, run_noisy_sweep


def test_zero_noise_point_has_no_failures():
    result = run_noisy_point(L=4, T=4, p=0.0, q=0.0, trials=5, seed=123)
    assert result == NoisySimulationResult(L=4, T=4, p=0.0, q=0.0, trials=5, failures=0)
    assert result.failure_rate == 0.0
    assert result.standard_error == 0.0


def test_noisy_sweep_is_reproducible_with_seed():
    first = run_noisy_sweep(L_values=[3, 4], p_values=[0.02, 0.03], trials=3, seed=7)
    second = run_noisy_sweep(L_values=[3, 4], p_values=[0.02, 0.03], trials=3, seed=7)
    assert first == second
    assert [item.T for item in first] == [3, 3, 4, 4]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_noisy.py -v`

Expected: FAIL with missing `NoisySimulationResult`, `run_noisy_point`, or `run_noisy_sweep`.

- [ ] **Step 3: Implement noisy Monte Carlo APIs**

Append to `src/toric_mwpm/noisy.py`:

```python
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


def _validate_run_parameters(L: int, T: int, p: float, q: float, trials: int) -> None:
    if L <= 0:
        raise ValueError("L must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if trials <= 0:
        raise ValueError("trials must be positive")
    _likelihood_weight(p)
    _likelihood_weight(q)


def _sample_noisy_history(
    *,
    L: int,
    T: int,
    p: float,
    q: float,
    rng: np.random.Generator,
) -> tuple[ToricChain, list[DetectionEvent]]:
    cumulative = ToricChain.empty(L)
    true_syndromes: list[np.ndarray] = [np.zeros((L, L), dtype=bool)]
    measurement_flips: list[np.ndarray] = []
    for _ in range(T):
        cumulative = cumulative + random_error(L, p, rng)
        true_syndromes.append(cumulative.syndrome())
        measurement_flips.append(rng.random((L, L)) < q)
    measured = measured_syndrome_history(true_syndromes, measurement_flips)
    measured.append(cumulative.syndrome())
    events = detection_events_from_measurements(measured)
    return cumulative, events


def run_noisy_point(
    *,
    L: int,
    T: int | None = None,
    p: float,
    q: float | None = None,
    trials: int,
    seed: int | None = None,
) -> NoisySimulationResult:
    T = L if T is None else T
    q = p if q is None else q
    _validate_run_parameters(L, T, p, q, trials)
    rng = np.random.default_rng(seed)
    failures = 0
    for _ in range(trials):
        total_error, events = _sample_noisy_history(L=L, T=T, p=p, q=q, rng=rng)
        recovery = decode_spacetime_mwpm(events, L=L, p=p, q=q)
        if (total_error + recovery).is_logical_failure():
            failures += 1
    return NoisySimulationResult(L=L, T=T, p=p, q=q, trials=trials, failures=failures)


def run_noisy_sweep(
    *,
    L_values: Iterable[int],
    p_values: Iterable[float],
    trials: int,
    q: float | None = None,
    T: int | None = None,
    seed: int | None = None,
) -> list[NoisySimulationResult]:
    L_values = list(L_values)
    p_values = list(p_values)
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(len(L_values) * len(p_values))
    results: list[NoisySimulationResult] = []
    index = 0
    for L in L_values:
        for p in p_values:
            local_q = p if q is None else q
            local_T = L if T is None else T
            results.append(
                run_noisy_point(
                    L=L,
                    T=local_T,
                    p=p,
                    q=local_q,
                    trials=trials,
                    seed=int(child_seeds[index].generate_state(1)[0]),
                )
            )
            index += 1
    return results
```

Modify `src/toric_mwpm/__init__.py` to export:

```python
from .noisy import NoisySimulationResult, run_noisy_point, run_noisy_sweep
```

- [ ] **Step 4: Run targeted tests**

Run: `python -m pytest tests/test_noisy.py -v`

Expected: all noisy tests pass.

- [ ] **Step 5: Run full regression suite**

Run: `python -m pytest -q`

Expected: all tests pass.

---

### Task 4: Noisy CSV and Plot Outputs

**Files:**
- Modify: `src/toric_mwpm/noisy.py`
- Test: `tests/test_noisy.py`

- [ ] **Step 1: Add failing tests for output schema and plot cleanup**

Append to `tests/test_noisy.py`:

```python
import csv

from toric_mwpm.noisy import plot_noisy_results, write_noisy_csv


def test_write_noisy_csv_schema(tmp_path):
    results = [NoisySimulationResult(L=4, T=4, p=0.02, q=0.02, trials=10, failures=1)]
    output = tmp_path / "noisy.csv"
    write_noisy_csv(results, output)

    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["L"] == "4"
    assert rows[0]["T"] == "4"
    assert rows[0]["p"] == "0.02"
    assert rows[0]["q"] == "0.02"
    assert rows[0]["trials"] == "10"
    assert rows[0]["failures"] == "1"
    assert "failure_rate" in rows[0]
    assert "standard_error" in rows[0]


def test_plot_noisy_results_writes_file_and_closes_figures(tmp_path):
    import matplotlib.pyplot as plt

    results = [
        NoisySimulationResult(L=3, T=3, p=0.02, q=0.02, trials=10, failures=0),
        NoisySimulationResult(L=3, T=3, p=0.03, q=0.03, trials=10, failures=1),
    ]
    output = tmp_path / "noisy.png"
    plot_noisy_results(results, output)

    assert output.exists()
    assert plt.get_fignums() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_noisy.py -v`

Expected: FAIL with missing `write_noisy_csv` and `plot_noisy_results`.

- [ ] **Step 3: Implement CSV and plotting helpers**

Append to `src/toric_mwpm/noisy.py`:

```python
def write_noisy_csv(results: Sequence[NoisySimulationResult], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "L",
                "T",
                "p",
                "q",
                "trials",
                "failures",
                "failure_rate",
                "standard_error",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "L": result.L,
                    "T": result.T,
                    "p": result.p,
                    "q": result.q,
                    "trials": result.trials,
                    "failures": result.failures,
                    "failure_rate": result.failure_rate,
                    "standard_error": result.standard_error,
                }
            )


def plot_noisy_results(results: Sequence[NoisySimulationResult], path: str | Path) -> None:
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    try:
        for L in sorted({result.L for result in results}):
            subset = sorted((result for result in results if result.L == L), key=lambda item: item.p)
            ax.errorbar(
                [item.p for item in subset],
                [item.failure_rate for item in subset],
                yerr=[item.standard_error for item in subset],
                marker="o",
                capsize=3,
                label=f"L={L}, T={subset[0].T if subset else L}",
            )
        ax.set_xlabel("data error probability p")
        ax.set_ylabel("logical failure rate")
        ax.set_title("Noisy syndrome spacetime MWPM")
        ax.grid(True, alpha=0.3)
        if results:
            ax.legend()
        fig.tight_layout()
        fig.savefig(output, dpi=150)
    finally:
        plt.close(fig)
```

- [ ] **Step 4: Run targeted tests**

Run: `python -m pytest tests/test_noisy.py -v`

Expected: all noisy tests pass.

- [ ] **Step 5: Run full regression suite**

Run: `python -m pytest -q`

Expected: all tests pass.

---

### Task 5: Noisy Threshold CLI

**Files:**
- Create: `scripts/run_noisy_threshold.py`
- Test: `tests/test_run_noisy_threshold.py`

- [ ] **Step 1: Write failing CLI smoke test**

Create `tests/test_run_noisy_threshold.py` with:

```python
import subprocess
import sys
from pathlib import Path


def test_run_noisy_threshold_cli_creates_outputs(tmp_path):
    csv_path = tmp_path / "noisy.csv"
    figure_path = tmp_path / "noisy.png"
    script = Path("scripts") / "run_noisy_threshold.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--sizes",
            "3,4",
            "--p-values",
            "0.02,0.03",
            "--trials",
            "2",
            "--seed",
            "11",
            "--csv",
            str(csv_path),
            "--figure",
            str(figure_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "completed 4 noisy parameter points" in completed.stdout
    assert csv_path.exists()
    assert figure_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_noisy_threshold.py -v`

Expected: FAIL because `scripts/run_noisy_threshold.py` does not exist.

- [ ] **Step 3: Implement CLI script**

Create `scripts/run_noisy_threshold.py` with:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toric_mwpm.noisy import plot_noisy_results, run_noisy_sweep, write_noisy_csv


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run noisy-syndrome 3D spacetime MWPM threshold sweeps.")
    parser.add_argument("--sizes", default="4,6,8", help="Comma-separated L values. Default: 4,6,8")
    parser.add_argument("--p-values", default="0.02,0.03,0.04,0.05", help="Comma-separated data error probabilities.")
    parser.add_argument("--q", type=float, default=None, help="Measurement error probability. Default: q=p for each point.")
    parser.add_argument("--T", type=int, default=None, help="Number of noisy measurement rounds. Default: T=L.")
    parser.add_argument("--trials", type=int, default=100, help="Monte Carlo trials per point.")
    parser.add_argument("--seed", type=int, default=20260522, help="Random seed.")
    parser.add_argument("--csv", default="outputs/noisy_threshold.csv", help="CSV output path.")
    parser.add_argument("--figure", default="outputs/noisy_threshold.png", help="Figure output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = run_noisy_sweep(
        L_values=_parse_int_list(args.sizes),
        p_values=_parse_float_list(args.p_values),
        q=args.q,
        T=args.T,
        trials=args.trials,
        seed=args.seed,
    )
    write_noisy_csv(results, args.csv)
    plot_noisy_results(results, args.figure)
    print(f"completed {len(results)} noisy parameter points")
    print(f"csv: {args.csv}")
    print(f"figure: {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI test**

Run: `python -m pytest tests/test_run_noisy_threshold.py -v`

Expected: 1 passed.

- [ ] **Step 5: Run full regression suite**

Run: `python -m pytest -q`

Expected: all tests pass.

---

### Task 6: README and Report Guidance

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add noisy-syndrome documentation**

Append this section to `README.md`:

```markdown
## Noisy syndrome / 3D spacetime extension

The simulator also includes a phenomenological noisy-syndrome model. Data errors occur on spatial toric-code edges with probability `p`, and each syndrome bit is independently flipped with measurement-error probability `q`. Repeated syndrome extraction produces detection events in spacetime `(x, y, t)`, which are decoded with a 3D MWPM graph. By default `q=p` and `T=L`.

Small smoke run:

```powershell
python scripts/run_noisy_threshold.py --sizes 3,4 --p-values 0.02,0.03 --trials 5 --csv outputs/noisy_smoke.csv --figure outputs/noisy_smoke.png
```

Report-quality exploratory run:

```powershell
python scripts/run_noisy_threshold.py --sizes 4,6,8 --p-values 0.015,0.02,0.025,0.03,0.035,0.04,0.05 --trials 500 --seed 20260522 --csv outputs/noisy_threshold.csv --figure outputs/noisy_threshold.png
```

Interpretation: this is a small-lattice spacetime MWPM demonstration of the threshold reduction caused by noisy measurement. It should be described qualitatively: perfect syndrome measurement gives a threshold scale near `p≈0.10`, while phenomenological noisy measurement shifts the threshold scale toward `p≈0.03`. Do not present small sweeps as a precision estimate of the Wang-Harrington-Preskill value `p_c0≈0.0293`.
```

- [ ] **Step 2: Run documentation smoke command**

Run: `python scripts/run_noisy_threshold.py --sizes 3 --p-values 0.02 --trials 2 --csv outputs/noisy_readme_smoke.csv --figure outputs/noisy_readme_smoke.png`

Expected: command exits 0 and creates both files.

- [ ] **Step 3: Run full regression suite**

Run: `python -m pytest -q`

Expected: all tests pass.

---

### Task 7: Final Review and Verification Sweep

**Files:**
- No required edits unless review finds issues.

- [ ] **Step 1: Run fresh full test suite**

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Run a noisy-syndrome report smoke sweep**

Run: `python scripts/run_noisy_threshold.py --sizes 4,6 --p-values 0.02,0.03,0.04 --trials 20 --seed 20260522 --csv outputs/noisy_report_smoke.csv --figure outputs/noisy_report_smoke.png`

Expected: command exits 0, reports 6 noisy parameter points, and creates both files.

- [ ] **Step 3: Request final code review**

Ask a reviewer to verify:

- existing 2D functionality still passes;
- noisy measurement history and final perfect boundary are correctly indexed;
- spacetime distance is spatially periodic and temporally open;
- recovery projection produces closed residual chains;
- CSV/plot/CLI outputs match documentation;
- README does not overclaim precision.

- [ ] **Step 4: Address any Critical or Important review issues**

If the reviewer reports issues, fix them and repeat Steps 1-3.

---

## Self-Review

Spec coverage:
- 3D noisy-syndrome model: Tasks 1 and 3.
- Detection events and final perfect boundary: Tasks 1 and 3.
- Spacetime MWPM and anisotropic distance: Task 2.
- Logical failure from total error plus recovery: Task 3.
- CSV/PNG outputs: Task 4.
- Separate CLI: Task 5.
- README/report interpretation: Task 6.
- Final verification and review: Task 7.

Placeholder scan:
- No placeholders, no TBDs, and no unspecified test steps remain.

Type consistency:
- Public names are consistent across tasks: `DetectionEvent`, `NoisySimulationResult`, `run_noisy_point`, `run_noisy_sweep`, `write_noisy_csv`, `plot_noisy_results`, `decode_spacetime_mwpm`, and `spacetime_distance`.
