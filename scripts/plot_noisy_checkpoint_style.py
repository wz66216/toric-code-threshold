from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _read_rows(csv_path: Path):
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                {
                    "L": int(row["L"]),
                    "p": float(row["p"]),
                    "failure_rate": float(row["failure_rate"]),
                }
            )
        return rows


def plot_checkpoint(csv_path: Path, figure_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _read_rows(csv_path)
    by_L = defaultdict(list)
    for row in rows:
        by_L[row["L"]].append(row)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    try:
        p_values = sorted({row["p"] for row in rows})
        ax.plot(p_values, p_values, linestyle=":", color="gray", linewidth=1.8, label=r"$P_{fail} = p$")
        for L in sorted(by_L):
            group = sorted(by_L[L], key=lambda row: row["p"])
            ax.plot([row["p"] for row in group], [row["failure_rate"] for row in group], marker="o", linewidth=1.8, markersize=4.5, label=f"L={L}")
        ax.set_xlabel("Physical error rate p")
        ax.set_ylabel("Logical failure rate")
        ax.set_title("Toric MWPM simulation")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--figure", required=True)
    args = parser.parse_args(argv)
    plot_checkpoint(Path(args.csv), Path(args.figure))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
