from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from matplotlib.figure import Figure


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


Z_95 = 1.959963984540054


def wilson_interval(failures: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if failures < 0 or failures > trials:
        raise ValueError("failures must be between 0 and trials")
    phat = failures / trials
    denom = 1.0 + z * z / trials
    center = (phat + z * z / (2 * trials)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4 * trials)) / trials) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, object | str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def validate_schema(rows: list[dict[str, str]], required: set[str], name: str) -> None:
    if not rows:
        raise ValueError(f"{name} has no data rows")
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def ensure_ci_columns(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    enriched: list[dict[str, str]] = []
    for row in rows:
        new_row = dict(row)
        needs_low = not new_row.get("ci95_low")
        needs_high = not new_row.get("ci95_high")
        if needs_low or needs_high:
            low, high = wilson_interval(int(new_row["failures"]), int(new_row["trials"]))
            if needs_low:
                new_row["ci95_low"] = str(low)
            if needs_high:
                new_row["ci95_high"] = str(high)
        enriched.append(new_row)
    return enriched


def add_q_ratio_column(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    enriched: list[dict[str, str]] = []
    for row in rows:
        new_row = dict(row)
        if not new_row.get("q_ratio") and "q" in new_row and "p" in new_row:
            p = float(new_row["p"])
            q = float(new_row["q"])
            new_row["q_ratio"] = "" if p == 0 else str(q / p)
        enriched.append(new_row)
    return enriched


def interpolate_pair_crossing(rows, L_small, L_large, experiment):
    by_L = {str(L_small): {}, str(L_large): {}}
    for row in rows:
        L = str(row.get("L"))
        if L in by_L and row.get("p") is not None:
            by_L[L][float(row["p"])] = row

    for L in (str(L_small), str(L_large)):
        for p, row in list(by_L[L].items()):
            by_L[L][p] = ensure_ci_columns([row])[0]

    common_ps = sorted(set(by_L[str(L_small)]).intersection(by_L[str(L_large)]))
    base = {
        "experiment": experiment,
        "L_pair": f"{L_small}-{L_large}",
        "crossing_method": "common_p_grid",
        "p_cross": "",
        "bracket_low_p": "",
        "bracket_high_p": "",
        "status": "no_common_grid",
        "trials_min": "",
        "ci_overlap_flag": "",
    }
    if not common_ps:
        return base

    diffs = []
    for p in common_ps:
        r_small = by_L[str(L_small)][p]
        r_large = by_L[str(L_large)][p]
        diff = float(r_small["failure_rate"]) - float(r_large["failure_rate"])
        diffs.append((p, diff, r_small, r_large))

    base["trials_min"] = str(min(min(int(r["trials"]) for r in (d[2], d[3])) for d in diffs))

    def ci_overlap(r_small: dict[str, str], r_large: dict[str, str]) -> str:
        try:
            low_s = float(r_small["ci95_low"])
            high_s = float(r_small["ci95_high"])
            low_l = float(r_large["ci95_low"])
            high_l = float(r_large["ci95_high"])
        except (KeyError, TypeError, ValueError):
            return "unknown"
        return "true" if max(low_s, low_l) <= min(high_s, high_l) else "false"

    def ci_overlap_at_p(p: float) -> str:
        r_small = by_L[str(L_small)].get(p)
        r_large = by_L[str(L_large)].get(p)
        if r_small is None or r_large is None:
            return "unknown"
        return ci_overlap(r_small, r_large)

    def ci_overlap_for_bracket(p1: float, p2: float) -> str:
        overlap1 = ci_overlap_at_p(p1)
        overlap2 = ci_overlap_at_p(p2)
        if overlap1 == "unknown" and overlap2 == "unknown":
            return "unknown"
        if overlap1 == "true" or overlap2 == "true":
            return "true"
        if overlap1 == "false" and overlap2 == "false":
            return "false"
        return overlap1 if overlap1 != "unknown" else overlap2

    for (p1, d1, _, _), (p2, d2, _, _) in zip(diffs, diffs[1:]):
        if d1 == 0:
            return {**base, "status": "bracketed", "p_cross": str(p1), "bracket_low_p": str(p1), "bracket_high_p": str(p1), "ci_overlap_flag": ci_overlap_at_p(p1)}
        if d1 * d2 < 0:
            p_cross = p1 + (0 - d1) * (p2 - p1) / (d2 - d1)
            return {**base, "status": "bracketed", "p_cross": str(p_cross), "bracket_low_p": str(p1), "bracket_high_p": str(p2), "ci_overlap_flag": ci_overlap_for_bracket(p1, p2)}

    last_p, last_diff, last_small, last_large = diffs[-1]
    if last_diff == 0:
        return {**base, "status": "bracketed", "p_cross": str(last_p), "bracket_low_p": str(last_p), "bracket_high_p": str(last_p), "ci_overlap_flag": ci_overlap_at_p(last_p)}

    nearest = min(diffs, key=lambda item: abs(item[1]))
    return {**base, "status": "nearest_gap_only", "p_cross": str(nearest[0]), "bracket_low_p": str(nearest[0]), "bracket_high_p": str(nearest[0]), "ci_overlap_flag": ci_overlap_at_p(nearest[0])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze report results and generate summary artifacts.")
    parser.add_argument("--perfect-csv", default=None, help="Perfect-threshold CSV")
    parser.add_argument("--noisy-csv", default=None, help="Noisy-threshold CSV")
    parser.add_argument("--sensitivity-csv", default=None, help="Measurement sensitivity CSV")
    parser.add_argument("--mismatch-csv", default=None, help="Decoder-mismatch CSV")
    parser.add_argument("--spacetime-figure", default=None, help="Spacetime detection-event figure path to record in the summary")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    return parser


def _git_sha() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _read_optional_csv(path_text: str | None) -> list[dict[str, str]]:
    if not path_text:
        return []
    return add_q_ratio_column(ensure_ci_columns(read_csv_rows(Path(path_text))))


def plot_threshold_curves(rows: list[dict[str, str]], output_path: Path, title: str) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    fig: Figure
    fig, ax = plt.subplots()
    try:
        by_L: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_L.setdefault(str(row.get("L")), []).append(row)
        for L, group in sorted(by_L.items(), key=lambda item: float(item[0])):
            group = sorted(group, key=lambda row: float(row["p"]))
            rates = [float(r["failure_rate"]) for r in group]
            ax.errorbar(
                [float(r["p"]) for r in group],
                rates,
                yerr=_asymmetric_ci_yerr(group, rates),
                marker="o",
                capsize=3,
                label=f"L={L}",
            )
        ax.set_title(title)
        ax.set_xlabel("p")
        ax.set_ylabel("failure_rate")
        ax.legend()
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    finally:
        plt.close(fig)


def plot_measurement_sensitivity(rows: list[dict[str, str]], output_path: Path) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    fig: Figure
    fig, ax = plt.subplots()
    try:
        by_ratio: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_ratio.setdefault(str(row.get("q_ratio", "")), []).append(row)
        for ratio, group in sorted(by_ratio.items(), key=lambda item: float(item[0])):
            group = sorted(group, key=lambda row: float(row["p"]))
            rates = [float(r["failure_rate"]) for r in group]
            ax.errorbar(
                [float(r["p"]) for r in group],
                rates,
                yerr=_asymmetric_ci_yerr(group, rates),
                marker="o",
                capsize=3,
                label=f"q/p={ratio}",
            )
        ax.set_title("Measurement-noise sensitivity")
        ax.set_xlabel("p")
        ax.set_ylabel("failure_rate")
        ax.legend()
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    finally:
        plt.close(fig)


def plot_decoder_mismatch(rows: list[dict[str, str]], output_path: Path) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    fig: Figure
    fig, ax = plt.subplots()
    try:
        by_ratio: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_ratio.setdefault(str(row.get("decoder_q_ratio", "")), []).append(row)
        for ratio, group in sorted(by_ratio.items(), key=lambda item: float(item[0])):
            group = sorted(group, key=lambda row: float(row["p_true"]))
            rates = [float(r["failure_rate"]) for r in group]
            ax.errorbar(
                [float(r["p_true"]) for r in group],
                rates,
                yerr=_asymmetric_ci_yerr(group, rates),
                marker="o",
                capsize=3,
                label=f"q_decode/q_true={ratio}",
            )
        ax.set_title("Decoder mismatch under noisy syndrome")
        ax.set_xlabel("true p")
        ax.set_ylabel("failure_rate")
        ax.legend()
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    finally:
        plt.close(fig)


def plot_threshold_comparison(perfect_rows: list[dict[str, str]], noisy_rows: list[dict[str, str]], output_path: Path) -> None:
    if not perfect_rows and not noisy_rows:
        return
    import matplotlib.pyplot as plt

    fig: Figure
    fig, ax = plt.subplots()
    try:
        if perfect_rows:
            for L, group in sorted(_group_by_L(perfect_rows).items(), key=lambda item: float(item[0])):
                group = sorted(group, key=lambda row: float(row["p"]))
                rates = [float(r["failure_rate"]) for r in group]
                ax.errorbar(
                    [float(r["p"]) for r in group],
                    rates,
                    yerr=_asymmetric_ci_yerr(group, rates),
                    marker="o",
                    linestyle="-",
                    capsize=2,
                    label=f"perfect L={L}",
                )
        if noisy_rows:
            for L, group in sorted(_group_by_L(noisy_rows).items(), key=lambda item: float(item[0])):
                group = sorted(group, key=lambda row: float(row["p"]))
                rates = [float(r["failure_rate"]) for r in group]
                ax.errorbar(
                    [float(r["p"]) for r in group],
                    rates,
                    yerr=_asymmetric_ci_yerr(group, rates),
                    marker="x",
                    linestyle="--",
                    capsize=2,
                    label=f"noisy L={L}",
                )
        ax.set_title("Perfect vs noisy threshold curves")
        ax.set_xlabel("p")
        ax.set_ylabel("failure_rate")
        ax.legend()
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    finally:
        plt.close(fig)


def _group_by_L(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("L")), []).append(row)
    return grouped


def _asymmetric_ci_yerr(rows: list[dict[str, str]], rates: list[float]) -> list[list[float]]:
    lows: list[float] = []
    highs: list[float] = []
    for row, rate in zip(rows, rates):
        low = float(row.get("ci95_low") or rate)
        high = float(row.get("ci95_high") or rate)
        lows.append(max(0.0, rate - low))
        highs.append(max(0.0, high - rate))
    return [lows, highs]


def _markdown_table(rows: list[dict[str, str]], fieldnames: list[str]) -> list[str]:
    if not rows:
        return []
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    return lines


def _format_float(text: str, digits: int = 4) -> str:
    if text == "":
        return ""
    try:
        return f"{float(text):.{digits}f}"
    except ValueError:
        return text


def _compact_crossings(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    compact = []
    for row in rows:
        compact.append({
            "experiment": row.get("experiment", ""),
            "L_pair": row.get("L_pair", ""),
            "p_cross": _format_float(row.get("p_cross", "")),
            "status": row.get("status", ""),
            "ci_overlap": row.get("ci_overlap_flag", ""),
        })
    return compact


def _sensitivity_summary(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("q_ratio", "")), []).append(row)
    means = []
    for ratio, group in grouped.items():
        rates = [float(row["failure_rate"]) for row in group]
        means.append((float(ratio), sum(rates) / len(rates)))
    means.sort()
    if len(means) < 2:
        return ""
    low_ratio, low_mean = means[0]
    high_ratio, high_mean = means[-1]
    return f"Across this fixed-size scan, increasing q/p from {low_ratio:g} to {high_ratio:g} changes the mean failure rate from {low_mean:.4f} to {high_mean:.4f}."


def _mismatch_summary(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("decoder_q_ratio", "")), []).append(row)
    means = []
    for ratio, group in grouped.items():
        rates = [float(row["failure_rate"]) for row in group]
        means.append((float(ratio), sum(rates) / len(rates)))
    means.sort()
    matched = next((mean for ratio, mean in means if abs(ratio - 1.0) < 1e-12), None)
    best_ratio, best_mean = min(means, key=lambda item: item[1])
    if matched is None:
        return f"The lowest mean failure rate in this scan occurs at q_decode/q_true={best_ratio:g} with mean failure rate {best_mean:.4f}."
    return f"The matched decoder q_decode/q_true=1 has mean failure rate {matched:.4f}; the lowest scanned mean is {best_mean:.4f} at q_decode/q_true={best_ratio:g}."


def write_summary_md(
    out_dir: Path,
    perfect_rows: list[dict[str, str]],
    noisy_rows: list[dict[str, str]],
    sensitivity_rows: list[dict[str, str]],
    mismatch_rows: list[dict[str, str]],
    crossings_rows: list[dict[str, str]],
    spacetime_figure: str | None,
) -> None:
    lines = [
        "# Analysis summary",
        "",
        "This report summarizes finite-size apparent crossings and related sensitivity scans.",
        "It is not a precision WHP reproduction claim.",
        "",
        "## Inputs",
        "",
        f"- perfect rows: {len(perfect_rows)}",
        f"- noisy rows: {len(noisy_rows)}",
        f"- measurement-sensitivity rows: {len(sensitivity_rows)}",
        f"- decoder-mismatch rows: {len(mismatch_rows)}",
        "",
    ]
    if crossings_rows:
        lines += [
            "## Crossings",
            "",
            f"- crossings computed: {len(crossings_rows)}",
            "",
            *_markdown_table(_compact_crossings(crossings_rows), ["experiment", "L_pair", "p_cross", "status", "ci_overlap"]),
            "",
        ]
    if sensitivity_rows:
        trend = _sensitivity_summary(sensitivity_rows)
        lines += [
            "## Measurement sensitivity",
            "",
            "The q-ratio scan is a fixed-size sensitivity check, not a threshold shift estimate.",
            *(["", trend] if trend else []),
            "",
        ]
    if mismatch_rows:
        trend = _mismatch_summary(mismatch_rows)
        lines += [
            "## Decoder mismatch",
            "",
            "This scan separates the true measurement-noise rate from the rate assumed by the decoder. It is a finite-size probe of decoder mismatch, or equivalently a practical deviation from the matched Nishimori weighting condition.",
            *(["", trend] if trend else []),
            "",
        ]
    if spacetime_figure:
        lines += [
            "## Spacetime visualization",
            "",
            f"- detection-event figure: `{spacetime_figure}`",
            "",
            "The 3D figure visualizes detection events and MWPM pairings in spacetime; the blue lines are pairing guides, not a full microscopic error-chain reconstruction.",
            "",
        ]
    lines += [
        "## Safe report wording",
        "",
        "The perfect-syndrome MWPM data should be compared with the zero-temperature/minimum-energy RBIM threshold scale, not with the full Nishimori-line maximum-likelihood threshold.",
        "",
        "The noisy-syndrome simulation shows a finite-size apparent crossing near the few-percent scale, broadly consistent with the expected measurement-error threshold scale, but it is not a precision finite-size-scaling estimate of the WHP threshold.",
        "",
        "The decoder-mismatch scan is best interpreted as evidence that practical decoding depends on how well the assumed likelihood weights match the true noise process.",
        "",
        "## Paper-ready Chinese wording",
        "",
        "在完全 syndrome 测量情形下，MWPM 解码对应零温或最小能量极限，因此本文数值结果主要与 WHP 给出的 RBIM 零温阈值进行比较。加入测量噪声后，重复 syndrome 测量引入时间维度，detection events 构成三维时空中的边界；本文的 noisy-syndrome MWPM 结果只作为有限尺寸量级验证，而不声称精密复现 3D RPGM 的相变点。",
        "",
        "decoder mismatch 实验进一步说明，统计力学映射中自然的 Nishimori 权重对应真实噪声与解码器假设相匹配；当解码器使用偏离真实噪声的权重时，逻辑失败率会反映这种模型失配的代价。",
        "",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_metadata(out_dir: Path, perfect_csv: str | None, noisy_csv: str | None, sensitivity_csv: str | None, mismatch_csv: str | None, spacetime_figure: str | None) -> None:
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "perfect_csv": perfect_csv,
        "noisy_csv": noisy_csv,
        "sensitivity_csv": sensitivity_csv,
        "mismatch_csv": mismatch_csv,
        "spacetime_figure": spacetime_figure,
        "git_sha": _git_sha(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _write_crossings(out_dir: Path, perfect_rows: list[dict[str, str]], noisy_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if perfect_rows:
        Ls = sorted({int(r["L"]) for r in perfect_rows})
        for L_small, L_large in zip(Ls, Ls[1:]):
            rows.append(interpolate_pair_crossing(perfect_rows, L_small, L_large, experiment="perfect"))
    if noisy_rows:
        Ls = sorted({int(r["L"]) for r in noisy_rows})
        for L_small, L_large in zip(Ls, Ls[1:]):
            rows.append(interpolate_pair_crossing(noisy_rows, L_small, L_large, experiment="noisy"))
    fieldnames = ["experiment", "L_pair", "crossing_method", "p_cross", "bracket_low_p", "bracket_high_p", "status", "trials_min", "ci_overlap_flag"]
    write_csv_rows(out_dir / "crossings.csv", rows, fieldnames)
    return rows


def main(argv=None):
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    perfect_rows = _read_optional_csv(args.perfect_csv)
    noisy_rows = _read_optional_csv(args.noisy_csv)
    sensitivity_rows = _read_optional_csv(args.sensitivity_csv)
    mismatch_rows = _read_optional_csv(args.mismatch_csv)
    crossings = _write_crossings(out_dir, perfect_rows, noisy_rows)
    plot_threshold_curves(noisy_rows, out_dir / "fig_noisy_crossing_fine.png", "Noisy threshold curves")
    plot_measurement_sensitivity(sensitivity_rows, out_dir / "fig_measurement_noise_sensitivity.png")
    plot_decoder_mismatch(mismatch_rows, out_dir / "fig_decoder_mismatch.png")
    plot_threshold_comparison(perfect_rows, noisy_rows, out_dir / "fig_threshold_comparison.png")
    write_summary_md(out_dir, perfect_rows, noisy_rows, sensitivity_rows, mismatch_rows, crossings, args.spacetime_figure)
    write_metadata(out_dir, args.perfect_csv, args.noisy_csv, args.sensitivity_csv, args.mismatch_csv, args.spacetime_figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
