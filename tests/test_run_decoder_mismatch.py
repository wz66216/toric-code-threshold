from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_decoder_mismatch_cli_writes_decoder_assumption_columns(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = tmp_path / "mismatch.csv"
    script = repo_root / "scripts" / "run_decoder_mismatch.py"

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
            "--true-q-ratio",
            "1.0",
            "--decoder-q-ratios",
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

    assert result.returncode == 0, result.stderr
    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert "q_decode" in lines[0]
    assert "decoder_q_ratio" in lines[0]
    assert len(lines) == 1 + 4
