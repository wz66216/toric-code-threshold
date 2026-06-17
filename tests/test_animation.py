from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from toric_mwpm.animation import generate_trial_frames, save_spacetime_detection_plot, save_trial_frames


def test_generate_trial_frames_returns_five_frames():
    frames = generate_trial_frames(4, 0.10, seed=42)
    assert len(frames) == 5
    for fig, title in frames:
        assert isinstance(title, str) and len(title) > 0
    # Close all to prevent leaks
    import matplotlib.pyplot as plt
    for fig, _ in frames:
        plt.close(fig)


def test_save_trial_frames_creates_png_files(tmp_path):
    paths = save_trial_frames(4, 0.10, 42, str(tmp_path), dpi=72)
    assert len(paths) == 5
    for p in paths:
        assert p.endswith(".png")
        assert Path(p).exists()
    assert len(list(tmp_path.glob("frame_*.png"))) == 5


def test_no_error_case_produces_empty_syndrome_and_success():
    frames = generate_trial_frames(4, 0.0, seed=0)
    import matplotlib.pyplot as plt
    titles = [title for _, title in frames]
    for fig, _ in frames:
        plt.close(fig)
    assert len(frames) == 5
    assert "0 defects" in titles[1]
    assert "SUCCESS" in titles[4]


def test_save_spacetime_detection_plot_creates_png(tmp_path):
    path = save_spacetime_detection_plot(4, 4, 0.08, 0.08, 42, tmp_path / "spacetime.png", dpi=72)

    assert path.endswith("spacetime.png")
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0
