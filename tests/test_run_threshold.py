from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_threshold_script_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = tmp_path / "smoke.csv"
    fig_path = tmp_path / "smoke.png"
    script = repo_root / "scripts" / "run_threshold.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--sizes",
            "4,6",
            "--p-values",
            "0.08,0.10,0.12",
            "--trials",
            "10",
            "--seed",
            "5",
            "--csv",
            str(csv_path),
            "--figure",
            str(fig_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "csv:" in result.stdout
    assert "figure:" in result.stdout
    assert "rough crossing estimate:" in result.stdout
    assert csv_path.exists()
    assert fig_path.exists()
