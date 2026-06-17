from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toric_mwpm.noisy import run_noisy_point


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run measurement-noise sensitivity experiments.")
    parser.add_argument("--L", type=int, required=True, help="Lattice size")
    parser.add_argument("--T", type=int, required=True, help="Number of noisy rounds")
    parser.add_argument("--p-values", required=True, help="Comma-separated physical error probabilities")
    parser.add_argument("--q-ratios", required=True, help="Comma-separated q/p ratios")
    parser.add_argument("--trials", type=int, default=100, help="Monte Carlo trials per point")
    parser.add_argument("--seed", type=int, default=20260522, help="Random seed")
    parser.add_argument("--csv", default="outputs/measurement_noise_sensitivity.csv", help="Output CSV path")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    p_values = _parse_float_list(args.p_values)
    q_ratios = _parse_float_list(args.q_ratios)
    seed_sequence = np.random.SeedSequence(args.seed)
    child_seeds = seed_sequence.spawn(len(p_values) * len(q_ratios))

    rows = []
    point_count = 0
    index = 0
    for p in p_values:
        for q_ratio in q_ratios:
            q = p * q_ratio
            if q >= 0.5:
                raise SystemExit(f"invalid q={q} for p={p} and q_ratio={q_ratio}; q must be < 0.5")
            point_seed = int(child_seeds[index].generate_state(1)[0])
            result = run_noisy_point(L=args.L, T=args.T, p=p, q=q, trials=args.trials, seed=point_seed)
            ci95_low, ci95_high = result.wilson_interval_95
            rows.append({
                "L": result.L,
                "T": result.T,
                "p": result.p,
                "q": result.q,
                "q_ratio": q_ratio,
                "trials": result.trials,
                "failures": result.failures,
                "failure_rate": result.failure_rate,
                "standard_error": result.standard_error,
                "ci95_low": ci95_low,
                "ci95_high": ci95_high,
            })
            point_count += 1
            index += 1

    output = Path(args.csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["L", "T", "p", "q", "q_ratio", "trials", "failures", "failure_rate", "standard_error", "ci95_low", "ci95_high"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"completed {point_count} measurement sensitivity points")
    print(f"csv: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
