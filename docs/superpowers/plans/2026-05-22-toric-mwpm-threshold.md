# Toric MWPM Threshold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Python simulation that estimates the toric-code MWPM logical-failure threshold near `p≈0.10` and generates original CSV/figure outputs for the course report.

**Architecture:** Implement a small, focused Python package under `src/toric_mwpm`. The lattice module owns toric-code chain algebra and homology checks; the decoder module owns MWPM; the simulation module owns Monte Carlo aggregation; a script provides a command-line interface.

**Tech Stack:** Python 3, `numpy`, `networkx`, `matplotlib`, `pytest`.

---

## File Structure

- Create `requirements.txt`: runtime/test dependencies.
- Create `src/toric_mwpm/__init__.py`: package exports.
- Create `src/toric_mwpm/lattice.py`: `ToricChain`, random errors, syndrome calculation, path construction, homology checks.
- Create `src/toric_mwpm/decoder.py`: complete-graph construction and MWPM recovery.
- Create `src/toric_mwpm/simulation.py`: Monte Carlo trials, sweeps, CSV writing, crossing estimate, plotting.
- Create `scripts/run_threshold.py`: CLI for quick and report-quality runs.
- Create `tests/test_lattice.py`: deterministic lattice/syndrome/homology tests.
- Create `tests/test_decoder.py`: decoder invariant tests.
- Create `tests/test_simulation.py`: smoke tests for the simulation layer.

No existing project code is modified because the repository currently contains research materials but no simulation source tree.

---

### Task 1: Project Skeleton and Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `src/toric_mwpm/__init__.py`
- Create: `tests/test_lattice.py`

- [ ] **Step 1: Write the initial failing import test**

Create `tests/test_lattice.py` with:

```python
from toric_mwpm.lattice import ToricChain


def test_empty_chain_has_expected_shape_and_no_edges():
    chain = ToricChain.empty(4)

    assert chain.L == 4
    assert chain.horizontal.shape == (4, 4)
    assert chain.vertical.shape == (4, 4)
    assert chain.weight() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lattice.py::test_empty_chain_has_expected_shape_and_no_edges -v`

Expected: FAIL with `ModuleNotFoundError` or `ImportError` because `toric_mwpm.lattice` does not exist yet.

- [ ] **Step 3: Add dependency file**

Create `requirements.txt` with:

```text
numpy>=1.24
networkx>=3.0
matplotlib>=3.7
pytest>=7.0
```

- [ ] **Step 4: Create package export**

Create `src/toric_mwpm/__init__.py` with:

```python
"""Toric-code MWPM threshold simulation package."""

from .lattice import ToricChain, random_error

__all__ = ["ToricChain", "random_error"]
```

- [ ] **Step 5: Implement minimal `ToricChain`**

Create `src/toric_mwpm/lattice.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ToricChain:
    """A mod-2 edge chain on an L x L toric square lattice.

    horizontal[x, y] is the edge from (x, y) to (x+1 mod L, y).
    vertical[x, y] is the edge from (x, y) to (x, y+1 mod L).
    """

    horizontal: np.ndarray
    vertical: np.ndarray

    def __post_init__(self) -> None:
        h = np.asarray(self.horizontal, dtype=bool)
        v = np.asarray(self.vertical, dtype=bool)
        if h.shape != v.shape:
            raise ValueError("horizontal and vertical arrays must have identical shape")
        if h.ndim != 2 or h.shape[0] != h.shape[1]:
            raise ValueError("chain arrays must be square two-dimensional arrays")
        object.__setattr__(self, "horizontal", h.copy())
        object.__setattr__(self, "vertical", v.copy())

    @property
    def L(self) -> int:
        return int(self.horizontal.shape[0])

    @classmethod
    def empty(cls, L: int) -> "ToricChain":
        if L <= 0:
            raise ValueError("L must be positive")
        return cls(np.zeros((L, L), dtype=bool), np.zeros((L, L), dtype=bool))

    def weight(self) -> int:
        return int(np.count_nonzero(self.horizontal) + np.count_nonzero(self.vertical))


def random_error(L: int, p: float, rng: np.random.Generator | None = None) -> ToricChain:
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    generator = np.random.default_rng() if rng is None else rng
    return ToricChain(generator.random((L, L)) < p, generator.random((L, L)) < p)
```

- [ ] **Step 6: Run the initial test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_lattice.py::test_empty_chain_has_expected_shape_and_no_edges -v`

Expected: PASS.

---

### Task 2: Syndrome and Homology Logic

**Files:**
- Modify: `src/toric_mwpm/lattice.py`
- Modify: `tests/test_lattice.py`

- [ ] **Step 1: Add failing tests for syndrome and logical loops**

Append to `tests/test_lattice.py`:

```python
import numpy as np


def test_single_horizontal_edge_has_two_syndrome_defects():
    chain = ToricChain.empty(5).with_horizontal(1, 2)

    defects = set(chain.defects())

    assert defects == {(1, 2), (2, 2)}
    assert chain.syndrome().sum() == 2


def test_nontrivial_horizontal_loop_has_empty_syndrome_and_x_winding():
    chain = ToricChain.empty(5)
    for x in range(5):
        chain = chain.with_horizontal(x, 0)

    assert chain.syndrome().sum() == 0
    assert chain.logical_parity() == (True, False)
    assert chain.is_logical_failure()


def test_mod2_addition_cancels_identical_chains():
    chain = ToricChain.empty(4).with_vertical(3, 1)

    combined = chain + chain

    assert combined.weight() == 0
    assert not np.any(combined.syndrome())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_lattice.py -v`

Expected: FAIL because `with_horizontal`, `with_vertical`, `syndrome`, `defects`, `logical_parity`, `is_logical_failure`, and `__add__` are not implemented.

- [ ] **Step 3: Implement chain operations**

Replace `src/toric_mwpm/lattice.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np

Vertex = tuple[int, int]


@dataclass(frozen=True)
class ToricChain:
    """A mod-2 edge chain on an L x L toric square lattice.

    horizontal[x, y] is the edge from (x, y) to (x+1 mod L, y).
    vertical[x, y] is the edge from (x, y) to (x, y+1 mod L).
    """

    horizontal: np.ndarray
    vertical: np.ndarray

    def __post_init__(self) -> None:
        h = np.asarray(self.horizontal, dtype=bool)
        v = np.asarray(self.vertical, dtype=bool)
        if h.shape != v.shape:
            raise ValueError("horizontal and vertical arrays must have identical shape")
        if h.ndim != 2 or h.shape[0] != h.shape[1]:
            raise ValueError("chain arrays must be square two-dimensional arrays")
        object.__setattr__(self, "horizontal", h.copy())
        object.__setattr__(self, "vertical", v.copy())

    @property
    def L(self) -> int:
        return int(self.horizontal.shape[0])

    @classmethod
    def empty(cls, L: int) -> "ToricChain":
        if L <= 0:
            raise ValueError("L must be positive")
        return cls(np.zeros((L, L), dtype=bool), np.zeros((L, L), dtype=bool))

    def copy_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return self.horizontal.copy(), self.vertical.copy()

    def with_horizontal(self, x: int, y: int) -> "ToricChain":
        h, v = self.copy_arrays()
        h[x % self.L, y % self.L] ^= True
        return ToricChain(h, v)

    def with_vertical(self, x: int, y: int) -> "ToricChain":
        h, v = self.copy_arrays()
        v[x % self.L, y % self.L] ^= True
        return ToricChain(h, v)

    def __add__(self, other: "ToricChain") -> "ToricChain":
        if self.L != other.L:
            raise ValueError("cannot add chains with different lattice sizes")
        return ToricChain(np.logical_xor(self.horizontal, other.horizontal), np.logical_xor(self.vertical, other.vertical))

    def weight(self) -> int:
        return int(np.count_nonzero(self.horizontal) + np.count_nonzero(self.vertical))

    def syndrome(self) -> np.ndarray:
        """Return vertex defects as a Boolean L x L array."""
        h_right = self.horizontal
        h_left = np.roll(self.horizontal, shift=1, axis=0)
        v_up = self.vertical
        v_down = np.roll(self.vertical, shift=1, axis=1)
        return np.logical_xor.reduce((h_right, h_left, v_up, v_down))

    def defects(self) -> list[Vertex]:
        xs, ys = np.nonzero(self.syndrome())
        return [(int(x), int(y)) for x, y in zip(xs, ys)]

    def logical_parity(self) -> tuple[bool, bool]:
        """Return parity across fixed torus cuts for x and y logical cycles."""
        x_winding = bool(np.count_nonzero(self.horizontal[self.L - 1, :]) % 2)
        y_winding = bool(np.count_nonzero(self.vertical[:, self.L - 1]) % 2)
        return x_winding, y_winding

    def is_logical_failure(self) -> bool:
        x_winding, y_winding = self.logical_parity()
        return x_winding or y_winding


def chain_from_edges(L: int, horizontal_edges: Iterable[Vertex] = (), vertical_edges: Iterable[Vertex] = ()) -> ToricChain:
    chain = ToricChain.empty(L)
    for x, y in horizontal_edges:
        chain = chain.with_horizontal(x, y)
    for x, y in vertical_edges:
        chain = chain.with_vertical(x, y)
    return chain


def random_error(L: int, p: float, rng: np.random.Generator | None = None) -> ToricChain:
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    generator = np.random.default_rng() if rng is None else rng
    return ToricChain(generator.random((L, L)) < p, generator.random((L, L)) < p)
```

- [ ] **Step 4: Run lattice tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_lattice.py -v`

Expected: PASS.

---

### Task 3: Toric Shortest Paths and MWPM Decoder

**Files:**
- Modify: `src/toric_mwpm/lattice.py`
- Create: `src/toric_mwpm/decoder.py`
- Create: `tests/test_decoder.py`

- [ ] **Step 1: Add failing decoder tests**

Create `tests/test_decoder.py` with:

```python
import numpy as np

from toric_mwpm.decoder import decode_mwpm, toric_distance
from toric_mwpm.lattice import ToricChain, shortest_path_chain


def test_toric_distance_uses_periodic_shortcut():
    assert toric_distance((0, 0), (4, 0), 5) == 1
    assert toric_distance((0, 0), (3, 3), 5) == 4


def test_shortest_path_has_requested_boundary():
    path = shortest_path_chain(5, (4, 1), (1, 1))

    assert set(path.defects()) == {(4, 1), (1, 1)}
    assert path.weight() == 2


def test_decoder_recovers_single_edge_error_trivially():
    error = ToricChain.empty(5).with_horizontal(2, 3)
    recovery = decode_mwpm(error.syndrome())
    closed = error + recovery

    assert np.array_equal(recovery.syndrome(), error.syndrome())
    assert closed.syndrome().sum() == 0
    assert not closed.is_logical_failure()


def test_decoder_recovery_matches_random_syndrome_boundary():
    rng = np.random.default_rng(123)
    for _ in range(25):
        error = ToricChain(rng.random((6, 6)) < 0.08, rng.random((6, 6)) < 0.08)
        recovery = decode_mwpm(error.syndrome())
        assert np.array_equal(recovery.syndrome(), error.syndrome())
```

- [ ] **Step 2: Run decoder tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_decoder.py -v`

Expected: FAIL because `decoder.py` and `shortest_path_chain` do not exist.

- [ ] **Step 3: Add shortest path construction to lattice module**

Append this function to `src/toric_mwpm/lattice.py`:

```python

def _signed_shortest_delta(start: int, end: int, L: int) -> int:
    forward = (end - start) % L
    backward = forward - L
    if abs(forward) <= abs(backward):
        return int(forward)
    return int(backward)


def shortest_path_chain(L: int, start: Vertex, end: Vertex) -> ToricChain:
    """Construct one deterministic shortest toric path between two vertices."""
    x, y = start
    target_x, target_y = end
    chain = ToricChain.empty(L)

    dx = _signed_shortest_delta(x, target_x, L)
    step_x = 1 if dx >= 0 else -1
    for _ in range(abs(dx)):
        if step_x == 1:
            chain = chain.with_horizontal(x, y)
            x = (x + 1) % L
        else:
            x = (x - 1) % L
            chain = chain.with_horizontal(x, y)

    dy = _signed_shortest_delta(y, target_y, L)
    step_y = 1 if dy >= 0 else -1
    for _ in range(abs(dy)):
        if step_y == 1:
            chain = chain.with_vertical(x, y)
            y = (y + 1) % L
        else:
            y = (y - 1) % L
            chain = chain.with_vertical(x, y)

    return chain
```

- [ ] **Step 4: Implement MWPM decoder**

Create `src/toric_mwpm/decoder.py` with:

```python
from __future__ import annotations

import networkx as nx
import numpy as np

from .lattice import ToricChain, Vertex, shortest_path_chain


def toric_distance(a: Vertex, b: Vertex, L: int) -> int:
    dx_raw = abs(a[0] - b[0])
    dy_raw = abs(a[1] - b[1])
    dx = min(dx_raw, L - dx_raw)
    dy = min(dy_raw, L - dy_raw)
    return int(dx + dy)


def _defects_from_syndrome(syndrome: np.ndarray) -> list[Vertex]:
    array = np.asarray(syndrome, dtype=bool)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("syndrome must be a square two-dimensional array")
    xs, ys = np.nonzero(array)
    defects = [(int(x), int(y)) for x, y in zip(xs, ys)]
    if len(defects) % 2 != 0:
        raise ValueError("a toric-code syndrome must contain an even number of defects")
    return defects


def decode_mwpm(syndrome: np.ndarray) -> ToricChain:
    """Decode a toric-code syndrome with minimum-weight perfect matching."""
    defects = _defects_from_syndrome(syndrome)
    L = int(np.asarray(syndrome).shape[0])
    if not defects:
        return ToricChain.empty(L)

    graph = nx.Graph()
    for index, defect in enumerate(defects):
        graph.add_node(index, defect=defect)

    for i in range(len(defects)):
        for j in range(i + 1, len(defects)):
            graph.add_edge(i, j, weight=toric_distance(defects[i], defects[j], L))

    matching = nx.algorithms.matching.min_weight_matching(graph, weight="weight")
    recovery = ToricChain.empty(L)
    for i, j in matching:
        path = shortest_path_chain(L, defects[i], defects[j])
        recovery = recovery + path

    return recovery
```

- [ ] **Step 5: Run decoder tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_decoder.py -v`

Expected: PASS.

---

### Task 4: Monte Carlo Simulation Layer

**Files:**
- Create: `src/toric_mwpm/simulation.py`
- Create: `tests/test_simulation.py`
- Modify: `src/toric_mwpm/__init__.py`

- [ ] **Step 1: Add failing simulation tests**

Create `tests/test_simulation.py` with:

```python
from toric_mwpm.simulation import SimulationResult, estimate_crossing, run_point, run_sweep


def test_run_point_returns_valid_rate():
    result = run_point(L=4, p=0.05, trials=20, seed=7)

    assert result.L == 4
    assert result.p == 0.05
    assert result.trials == 20
    assert 0 <= result.failures <= 20
    assert 0.0 <= result.failure_rate <= 1.0
    assert result.standard_error >= 0.0


def test_run_sweep_returns_all_parameter_combinations():
    results = run_sweep(L_values=[4, 6], p_values=[0.05, 0.10], trials=5, seed=11)

    assert len(results) == 4
    assert {(r.L, r.p) for r in results} == {(4, 0.05), (4, 0.10), (6, 0.05), (6, 0.10)}


def test_estimate_crossing_uses_nearest_curve_difference():
    results = [
        SimulationResult(L=4, p=0.09, trials=100, failures=40),
        SimulationResult(L=8, p=0.09, trials=100, failures=30),
        SimulationResult(L=4, p=0.10, trials=100, failures=45),
        SimulationResult(L=8, p=0.10, trials=100, failures=46),
        SimulationResult(L=4, p=0.11, trials=100, failures=50),
        SimulationResult(L=8, p=0.11, trials=100, failures=60),
    ]

    assert estimate_crossing(results) == 0.10
```

- [ ] **Step 2: Run simulation tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_simulation.py -v`

Expected: FAIL because `simulation.py` does not exist.

- [ ] **Step 3: Implement simulation module**

Create `src/toric_mwpm/simulation.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
from typing import Sequence

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
        return self.failures / self.trials if self.trials else math.nan

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
        closed_chain = error + recovery
        if closed_chain.is_logical_failure():
            failures += 1
    return SimulationResult(L=L, p=float(p), trials=int(trials), failures=int(failures))


def run_sweep(L_values: Sequence[int], p_values: Sequence[float], trials: int, seed: int | None = None) -> list[SimulationResult]:
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(len(L_values) * len(p_values))
    results: list[SimulationResult] = []
    seed_index = 0
    for L in L_values:
        for p in p_values:
            point_seed = int(child_seeds[seed_index].generate_state(1)[0])
            seed_index += 1
            results.append(run_point(int(L), float(p), int(trials), point_seed))
    return results


def write_csv(results: Sequence[SimulationResult], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["L", "p", "trials", "failures", "failure_rate", "standard_error"])
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "L": result.L,
                    "p": result.p,
                    "trials": result.trials,
                    "failures": result.failures,
                    "failure_rate": result.failure_rate,
                    "standard_error": result.standard_error,
                }
            )


def estimate_crossing(results: Sequence[SimulationResult]) -> float | None:
    sizes = sorted({result.L for result in results})
    if len(sizes) < 2:
        return None
    low_L, high_L = sizes[0], sizes[-1]
    by_key = {(result.L, result.p): result.failure_rate for result in results}
    common_p = sorted({result.p for result in results if (low_L, result.p) in by_key and (high_L, result.p) in by_key})
    if not common_p:
        return None
    return min(common_p, key=lambda p: abs(by_key[(low_L, p)] - by_key[(high_L, p)]))


def plot_results(results: Sequence[SimulationResult], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for L in sorted({result.L for result in results}):
        subset = sorted((result for result in results if result.L == L), key=lambda r: r.p)
        ps = [result.p for result in subset]
        rates = [result.failure_rate for result in subset]
        errors = [result.standard_error for result in subset]
        ax.errorbar(ps, rates, yerr=errors, marker="o", capsize=3, label=f"L={L}")
    crossing = estimate_crossing(results)
    if crossing is not None:
        ax.axvline(crossing, color="black", linestyle="--", linewidth=1, alpha=0.6, label=f"rough crossing ≈ {crossing:.3f}")
    ax.set_xlabel("Physical bit-flip probability p")
    ax.set_ylabel("Logical failure probability")
    ax.set_title("Toric code MWPM threshold reproduction")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
```

- [ ] **Step 4: Update package exports**

Replace `src/toric_mwpm/__init__.py` with:

```python
"""Toric-code MWPM threshold simulation package."""

from .lattice import ToricChain, random_error
from .simulation import SimulationResult, run_point, run_sweep

__all__ = ["SimulationResult", "ToricChain", "random_error", "run_point", "run_sweep"]
```

- [ ] **Step 5: Run simulation tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_simulation.py -v`

Expected: PASS.

---

### Task 5: Command-Line Script and End-to-End Outputs

**Files:**
- Create: `scripts/run_threshold.py`

- [ ] **Step 1: Create CLI script**

Create `scripts/run_threshold.py` with:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from toric_mwpm.simulation import estimate_crossing, plot_results, run_sweep, write_csv


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run toric-code MWPM threshold simulation.")
    parser.add_argument("--sizes", default="6,8,10", help="Comma-separated lattice sizes, e.g. 8,12,16")
    parser.add_argument(
        "--p-values",
        default="0.07,0.08,0.09,0.10,0.11,0.12,0.13",
        help="Comma-separated physical error probabilities",
    )
    parser.add_argument("--trials", type=int, default=100, help="Monte Carlo trials per (L, p) point")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--csv", default="outputs/toric_mwpm_threshold.csv", help="Output CSV path")
    parser.add_argument("--figure", default="outputs/toric_mwpm_threshold.png", help="Output figure path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sizes = _parse_int_list(args.sizes)
    p_values = _parse_float_list(args.p_values)
    results = run_sweep(sizes, p_values, args.trials, args.seed)
    write_csv(results, Path(args.csv))
    plot_results(results, Path(args.figure))
    crossing = estimate_crossing(results)
    print(f"completed {len(results)} parameter points")
    print(f"csv: {args.csv}")
    print(f"figure: {args.figure}")
    if crossing is not None:
        print(f"rough crossing estimate: p ≈ {crossing:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all tests**

Run: `PYTHONPATH=src python -m pytest -v`

Expected: PASS for all tests.

- [ ] **Step 3: Run a quick end-to-end simulation**

Run: `PYTHONPATH=src python scripts/run_threshold.py --sizes 4,6 --p-values 0.08,0.10,0.12 --trials 10 --seed 5 --csv outputs/smoke_threshold.csv --figure outputs/smoke_threshold.png`

Expected: command exits with status 0 and prints paths plus a rough crossing estimate. Files `outputs/smoke_threshold.csv` and `outputs/smoke_threshold.png` exist.

---

### Task 6: Documentation and Report Notes

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create usage documentation**

Create `README.md` with:

```markdown
# Toric Code MWPM Threshold Reproduction

This project numerically reproduces the two-dimensional toric-code minimum-weight decoding threshold under independent bit-flip noise and perfect syndrome measurement.

The simulation generates random error chains, computes the syndrome, decodes with minimum-weight perfect matching, and checks whether the combined error-plus-recovery chain is homologically nontrivial. The resulting logical-failure-rate curves cross near `p≈0.10`, consistent with the zero-temperature/minimum-energy RBIM threshold reported by Wang, Harrington, and Preskill.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run tests

```bash
PYTHONPATH=src python -m pytest -v
```

On Windows PowerShell, use:

```powershell
$env:PYTHONPATH="src"; python -m pytest -v
```

## Quick simulation

```bash
PYTHONPATH=src python scripts/run_threshold.py --sizes 6,8,10 --trials 100
```

On Windows PowerShell, use:

```powershell
$env:PYTHONPATH="src"; python scripts/run_threshold.py --sizes 6,8,10 --trials 100
```

Outputs are written to `outputs/toric_mwpm_threshold.csv` and `outputs/toric_mwpm_threshold.png` by default.

## Report interpretation

MWPM chooses a minimum-weight recovery chain, so it corresponds to the zero-temperature/minimum-energy limit of the RBIM decoder. It should be compared with the reported `p_c0≈0.1031` value rather than the full Nishimori-line maximum-likelihood threshold near `p≈0.109`.

The simulation supports the course report's statistical-mechanical interpretation: local syndrome constraints define a boundary, while logical failure is determined by the homology class of the closed chain `E+R`.
```

- [ ] **Step 2: Run documentation smoke command**

Run: `$env:PYTHONPATH="src"; python scripts/run_threshold.py --sizes 4 --p-values 0.10 --trials 2 --csv outputs/readme_smoke.csv --figure outputs/readme_smoke.png`

Expected: command exits with status 0 and creates both output files.

---

## Self-Review Checklist

- Spec coverage: The plan creates source code, tests, CLI, CSV output, figure output, and documentation for reproducing the Toric MWPM threshold chosen in the design document.
- TDD coverage: Tasks 1-4 write failing tests before implementation. Task 5 validates end-to-end execution. Task 6 documents usage and re-runs a smoke command.
- YAGNI: Full finite-temperature RBIM Monte Carlo, full ML decoding, 3D RPGM, color-code, and fracton-code simulations are intentionally excluded.
- Type consistency: The exported functions are `ToricChain`, `random_error`, `decode_mwpm`, `SimulationResult`, `run_point`, and `run_sweep`; these names are consistent across tests, package exports, and CLI.
- Verification commands: The final required commands are `PYTHONPATH=src python -m pytest -v` and the quick `scripts/run_threshold.py` smoke run.
