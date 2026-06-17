from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toric_mwpm.animation import save_spacetime_detection_plot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot one noisy-syndrome spacetime detection-event trial.")
    parser.add_argument("--L", type=int, default=6, help="Lattice size")
    parser.add_argument("--T", type=int, default=6, help="Number of noisy rounds")
    parser.add_argument("--p", type=float, default=0.03, help="True data error probability")
    parser.add_argument("--q", type=float, default=0.03, help="True measurement error probability")
    parser.add_argument("--decoder-p", type=float, default=None, help="Decoder-assumed data error probability")
    parser.add_argument("--decoder-q", type=float, default=None, help="Decoder-assumed measurement error probability")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--figure", default="outputs/spacetime_detection.png", help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    path = save_spacetime_detection_plot(
        args.L,
        args.T,
        args.p,
        args.q,
        args.seed,
        args.figure,
        decoder_p=args.decoder_p,
        decoder_q=args.decoder_q,
        dpi=args.dpi,
    )
    print(f"figure: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
