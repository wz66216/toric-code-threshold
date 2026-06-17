from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toric_mwpm.noisy import run_noisy_point


FIELDNAMES = ["L", "T", "p", "q", "trials", "failures", "failure_rate", "standard_error", "ci95_low", "ci95_high"]


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _read_completed(path: Path) -> set[tuple[int, float]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {(int(row["L"]), float(row["p"])) for row in csv.DictReader(handle)}


def _run_point(payload):
    L, p, trials, seed = payload
    result = run_noisy_point(L=L, T=L, p=p, q=p, trials=trials, seed=seed)
    ci95_low, ci95_high = result.wilson_interval_95
    return {
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
    }


def _append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def _sort_csv(path: Path) -> None:
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: (int(row["L"]), float(row["p"])))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run noisy threshold sweep with per-point checkpointing.")
    parser.add_argument("--sizes", required=True, help="Comma-separated lattice sizes")
    parser.add_argument("--p-values", required=True, help="Comma-separated physical error probabilities")
    parser.add_argument("--trials", type=int, default=1000, help="Monte Carlo trials per point")
    parser.add_argument("--seed", type=int, default=20260610, help="Random seed")
    parser.add_argument("--csv", required=True, help="Checkpoint/output CSV path")
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel parameter points")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    sizes = _parse_int_list(args.sizes)
    p_values = _parse_float_list(args.p_values)
    output = Path(args.csv)
    completed = _read_completed(output)
    all_points = [(L, p) for L in sizes for p in p_values]
    pending = [(L, p) for L, p in all_points if (L, p) not in completed]
    seed_sequence = np.random.SeedSequence(args.seed)
    child_seeds = seed_sequence.spawn(len(all_points))
    seed_by_point = {point: int(child_seeds[index].generate_state(1)[0]) for index, point in enumerate(all_points)}
    payloads = [(L, p, args.trials, seed_by_point[(L, p)]) for L, p in pending]

    print(f"completed existing points: {len(completed)}")
    print(f"pending points: {len(payloads)}")
    if not payloads:
        _sort_csv(output)
        print(f"csv: {output}")
        return 0

    finished_now = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_point = {executor.submit(_run_point, payload): (payload[0], payload[1]) for payload in payloads}
        for future in as_completed(future_to_point):
            L, p = future_to_point[future]
            row = future.result()
            _append_row(output, row)
            finished_now += 1
            print(f"finished {finished_now}/{len(payloads)}: L={L}, p={p:.3f}, failure_rate={row['failure_rate']:.6f}", flush=True)

    _sort_csv(output)
    print(f"csv: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
