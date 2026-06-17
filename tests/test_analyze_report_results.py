import importlib.util
from pathlib import Path


def load_analysis_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_report_results.py"
    spec = importlib.util.spec_from_file_location("analyze_report_results", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_import_does_not_force_matplotlib_backend(monkeypatch):
    import matplotlib

    def fail_use(*args, **kwargs):
        raise AssertionError("matplotlib.use() should not be called at import time")

    monkeypatch.setattr(matplotlib, "use", fail_use)
    load_analysis_module()


def test_wilson_columns_are_added_for_rows_without_ci():
    analysis = load_analysis_module()
    rows = [{"L": "4", "p": "0.03", "trials": "100", "failures": "10", "failure_rate": "0.1"}]
    enriched = analysis.ensure_ci_columns(rows)
    assert "ci95_low" in enriched[0]
    assert "ci95_high" in enriched[0]
    assert 0.0 <= float(enriched[0]["ci95_low"]) <= 0.1
    assert 0.1 <= float(enriched[0]["ci95_high"]) <= 1.0


def test_ensure_ci_columns_preserves_existing_single_bound():
    analysis = load_analysis_module()
    rows = [{"L": "4", "p": "0.03", "trials": "100", "failures": "10", "failure_rate": "0.1", "ci95_low": "0.05"}]
    enriched = analysis.ensure_ci_columns(rows)
    assert enriched[0]["ci95_low"] == "0.05"
    assert enriched[0]["ci95_high"] != ""


def test_interpolate_pair_crossing_reports_bracketed_sign_change():
    analysis = load_analysis_module()
    rows = [
        {"L": "4", "p": "0.02", "failure_rate": "0.20", "trials": "100", "failures": "20"},
        {"L": "4", "p": "0.04", "failure_rate": "0.30", "trials": "100", "failures": "30"},
        {"L": "6", "p": "0.02", "failure_rate": "0.10", "trials": "100", "failures": "10"},
        {"L": "6", "p": "0.04", "failure_rate": "0.40", "trials": "100", "failures": "40"},
    ]
    result = analysis.interpolate_pair_crossing(rows, 4, 6, experiment="toy")
    assert result["status"] == "bracketed"
    assert float(result["bracket_low_p"]) == 0.02
    assert float(result["bracket_high_p"]) == 0.04
    assert 0.02 < float(result["p_cross"]) < 0.04


def test_interpolate_pair_crossing_reports_bracketed_on_exact_zero_at_last_common_p():
    analysis = load_analysis_module()
    rows = [
        {"L": "4", "p": "0.02", "failure_rate": "0.20", "trials": "100", "failures": "20"},
        {"L": "4", "p": "0.04", "failure_rate": "0.30", "trials": "100", "failures": "30"},
        {"L": "6", "p": "0.02", "failure_rate": "0.10", "trials": "100", "failures": "10"},
        {"L": "6", "p": "0.04", "failure_rate": "0.30", "trials": "100", "failures": "30"},
    ]
    result = analysis.interpolate_pair_crossing(rows, 4, 6, experiment="toy")
    assert result["status"] == "bracketed"
    assert float(result["p_cross"]) == 0.04
    assert float(result["bracket_low_p"]) == 0.04
    assert float(result["bracket_high_p"]) == 0.04


def test_interpolate_pair_crossing_sets_ci_overlap_flag_for_overlap_and_non_overlap():
    analysis = load_analysis_module()
    overlap_rows = [
        {"L": "4", "p": "0.02", "failure_rate": "0.10", "trials": "100", "failures": "10"},
        {"L": "6", "p": "0.02", "failure_rate": "0.12", "trials": "100", "failures": "12"},
        {"L": "4", "p": "0.04", "failure_rate": "0.20", "trials": "100", "failures": "20"},
        {"L": "6", "p": "0.04", "failure_rate": "0.18", "trials": "100", "failures": "18"},
    ]
    non_overlap_rows = [
        {"L": "4", "p": "0.02", "failure_rate": "0.05", "trials": "100", "failures": "5"},
        {"L": "6", "p": "0.02", "failure_rate": "0.30", "trials": "100", "failures": "30"},
        {"L": "4", "p": "0.04", "failure_rate": "0.06", "trials": "100", "failures": "6"},
        {"L": "6", "p": "0.04", "failure_rate": "0.35", "trials": "100", "failures": "35"},
    ]
    assert analysis.interpolate_pair_crossing(overlap_rows, 4, 6, experiment="toy")["ci_overlap_flag"] == "true"
    assert analysis.interpolate_pair_crossing(non_overlap_rows, 4, 6, experiment="toy")["ci_overlap_flag"] == "false"


def test_interpolate_pair_crossing_uses_selected_bracket_for_ci_overlap_flag():
    analysis = load_analysis_module()
    rows = [
        {"L": "4", "p": "0.01", "failure_rate": "0.05", "trials": "100", "failures": "5"},
        {"L": "6", "p": "0.01", "failure_rate": "0.30", "trials": "100", "failures": "30"},
        {"L": "4", "p": "0.02", "failure_rate": "0.52", "trials": "100", "failures": "52"},
        {"L": "6", "p": "0.02", "failure_rate": "0.50", "trials": "100", "failures": "50"},
    ]
    result = analysis.interpolate_pair_crossing(rows, 4, 6, experiment="toy")
    assert result["status"] == "bracketed"
    assert result["ci_overlap_flag"] == "true"


def test_interpolate_pair_crossing_reports_nearest_gap_without_sign_change():
    analysis = load_analysis_module()
    rows = [
        {"L": "4", "p": "0.02", "failure_rate": "0.20", "trials": "100", "failures": "20"},
        {"L": "4", "p": "0.04", "failure_rate": "0.25", "trials": "100", "failures": "25"},
        {"L": "6", "p": "0.02", "failure_rate": "0.10", "trials": "100", "failures": "10"},
        {"L": "6", "p": "0.04", "failure_rate": "0.15", "trials": "100", "failures": "15"},
    ]
    result = analysis.interpolate_pair_crossing(rows, 4, 6, experiment="toy")
    assert result["status"] == "nearest_gap_only"
    assert result["p_cross"] != ""


def test_analysis_cli_writes_summary_crossings_and_figures(tmp_path):
    import subprocess
    import sys

    perfect = tmp_path / "perfect.csv"
    noisy = tmp_path / "noisy.csv"
    sensitivity = tmp_path / "sensitivity.csv"
    mismatch = tmp_path / "mismatch.csv"
    out_dir = tmp_path / "report"

    perfect.write_text(
        "L,p,trials,failures,failure_rate,standard_error\n"
        "4,0.08,20,2,0.1,0.0\n"
        "4,0.12,20,8,0.4,0.0\n"
        "6,0.08,20,1,0.05,0.0\n"
        "6,0.12,20,10,0.5,0.0\n",
        encoding="utf-8",
    )
    noisy.write_text(
        "L,T,p,q,trials,failures,failure_rate,standard_error\n"
        "4,4,0.03,0.03,20,4,0.2,0.0\n"
        "4,4,0.04,0.04,20,8,0.4,0.0\n"
        "6,6,0.03,0.03,20,2,0.1,0.0\n"
        "6,6,0.04,0.04,20,10,0.5,0.0\n",
        encoding="utf-8",
    )
    sensitivity.write_text(
        "L,T,p,q,q_ratio,trials,failures,failure_rate,standard_error\n"
        "6,6,0.03,0.015,0.5,20,2,0.1,0.0\n"
        "6,6,0.03,0.03,1.0,20,4,0.2,0.0\n"
        "6,6,0.03,0.06,2.0,20,6,0.3,0.0\n",
        encoding="utf-8",
    )
    mismatch.write_text(
        "L,T,p_true,q_true,true_q_ratio,p_decode,q_decode,decoder_q_ratio,trials,failures,failure_rate,standard_error\n"
        "6,6,0.03,0.03,1.0,0.03,0.015,0.5,20,5,0.25,0.0\n"
        "6,6,0.03,0.03,1.0,0.03,0.03,1.0,20,3,0.15,0.0\n"
        "6,6,0.03,0.03,1.0,0.03,0.06,2.0,20,6,0.3,0.0\n",
        encoding="utf-8",
    )

    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_report_results.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--perfect-csv",
            str(perfect),
            "--noisy-csv",
            str(noisy),
            "--sensitivity-csv",
            str(sensitivity),
            "--mismatch-csv",
            str(mismatch),
            "--spacetime-figure",
            "outputs/report/spacetime_detection.png",
            "--out-dir",
            str(out_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / "crossings.csv").exists()
    assert (out_dir / "report_summary.md").exists()
    assert (out_dir / "analysis_metadata.json").exists()
    assert (out_dir / "fig_noisy_crossing_fine.png").exists()
    assert (out_dir / "fig_measurement_noise_sensitivity.png").exists()
    assert (out_dir / "fig_decoder_mismatch.png").exists()
    assert (out_dir / "fig_threshold_comparison.png").exists()
    summary = (out_dir / "report_summary.md").read_text(encoding="utf-8")
    assert "Decoder mismatch" in summary
    assert "Paper-ready Chinese wording" in summary
    metadata = (out_dir / "analysis_metadata.json").read_text(encoding="utf-8")
    assert "mismatch_csv" in metadata
    assert "spacetime_figure" in metadata
