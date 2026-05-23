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
    parser = argparse.ArgumentParser(description="Run noisy toric-code threshold simulation.")
    parser.add_argument("--sizes", default="4,6,8", help="Comma-separated lattice sizes")
    parser.add_argument("--p-values", default="0.02,0.03,0.04,0.05", help="Comma-separated physical error probabilities")
    parser.add_argument("--q", type=float, default=None, help="Measurement error probability (defaults to p)")
    parser.add_argument("--T", type=int, default=None, help="Number of noisy rounds (defaults to L)")
    parser.add_argument("--trials", type=int, default=100, help="Monte Carlo trials per point")
    parser.add_argument("--seed", type=int, default=20260522, help="Random seed")
    parser.add_argument("--csv", default="outputs/noisy_threshold.csv", help="Output CSV path")
    parser.add_argument("--figure", default="outputs/noisy_threshold.png", help="Output figure path")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    sizes = _parse_int_list(args.sizes)
    p_values = _parse_float_list(args.p_values)
    results = run_noisy_sweep(L_values=sizes, p_values=p_values, trials=args.trials, q=args.q, T=args.T, seed=args.seed)
    write_noisy_csv(results, Path(args.csv))
    plot_noisy_results(results, Path(args.figure))
    print(f"completed {len(results)} noisy parameter points")
    print(f"csv: {args.csv}")
    print(f"figure: {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
