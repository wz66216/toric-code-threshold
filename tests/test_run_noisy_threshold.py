from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_noisy_threshold_script_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = tmp_path / "noisy.csv"
    fig_path = tmp_path / "noisy.png"
    script = repo_root / "scripts" / "run_noisy_threshold.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--sizes",
            "3,4",
            "--p-values",
            "0.02,0.03",
            "--trials",
            "2",
            "--seed",
            "11",
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
    assert "completed 4 noisy parameter points" in result.stdout
    assert csv_path.exists()
    assert fig_path.exists()
