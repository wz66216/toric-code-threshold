from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_measurement_sensitivity_cli_writes_q_ratio_csv(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = tmp_path / "sensitivity.csv"
    script = repo_root / "scripts" / "run_measurement_sensitivity.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--L",
            "4",
            "--T",
            "4",
            "--p-values",
            "0.02,0.03",
            "--q-ratios",
            "0.5,1.0",
            "--trials",
            "2",
            "--seed",
            "11",
            "--csv",
            str(csv_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert csv_path.exists()
    content = csv_path.read_text(encoding="utf-8").splitlines()
    assert "q_ratio" in content[0]
    assert len(content) == 1 + 4
