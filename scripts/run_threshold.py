from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
