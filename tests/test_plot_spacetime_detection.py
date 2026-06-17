from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_plot_spacetime_detection_cli_writes_png(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    fig_path = tmp_path / "spacetime.png"
    script = repo_root / "scripts" / "plot_spacetime_detection.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--L",
            "4",
            "--T",
            "4",
            "--p",
            "0.08",
            "--q",
            "0.08",
            "--seed",
            "42",
            "--figure",
            str(fig_path),
            "--dpi",
            "72",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "figure:" in result.stdout
    assert fig_path.exists()
    assert fig_path.stat().st_size > 0
