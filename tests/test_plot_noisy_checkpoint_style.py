from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


def load_plot_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "plot_noisy_checkpoint_style.py"
    spec = importlib.util.spec_from_file_location("plot_noisy_checkpoint_style", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_plot_noisy_checkpoint_style_cli_uses_reference_layout(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt

    main = load_plot_module().main

    csv_path = tmp_path / "checkpoint.csv"
    figure_path = tmp_path / "checkpoint.png"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["L", "T", "p", "q", "trials", "failures", "failure_rate", "standard_error", "ci95_low", "ci95_high"])
        writer.writerow([4, 4, 0.03, 0.03, 10, 1, 0.1, 0.03, 0.02, 0.18])
        writer.writerow([3, 4, 0.01, 0.01, 10, 0, 0.0, 0.0, 0.0, 0.0])
        writer.writerow([4, 4, 0.02, 0.02, 10, 2, 0.2, 0.04, 0.08, 0.34])

    captured = {}

    class FakeAxes:
        def plot(self, x, y, **kwargs):
            captured.setdefault("plot", []).append((list(x), list(y), kwargs))

        def set_xlabel(self, value):
            captured["xlabel"] = value

        def set_ylabel(self, value):
            captured["ylabel"] = value

        def set_title(self, value):
            captured["title"] = value

        def grid(self, *args, **kwargs):
            captured["grid"] = (args, kwargs)

        def legend(self, *args, **kwargs):
            captured["legend"] = (args, kwargs)

    class FakeFigure:
        def savefig(self, path, **kwargs):
            captured["savefig"] = (path, kwargs)

    fake_fig = FakeFigure()
    fake_ax = FakeAxes()
    monkeypatch.setattr(plt, "subplots", lambda *args, **kwargs: (fake_fig, fake_ax))
    monkeypatch.setattr(plt, "close", lambda fig: captured.setdefault("closed", fig))

    main(["--csv", str(csv_path), "--figure", str(figure_path)])

    assert [entry[2]["label"] for entry in captured["plot"]] == [r"$P_{fail} = p$", "L=3", "L=4"]
    assert captured["plot"][0][0] == [0.01, 0.02, 0.03]
    assert captured["plot"][0][1] == [0.01, 0.02, 0.03]
    assert captured["plot"][0][2]["linestyle"] == ":"
    assert captured["plot"][0][2]["color"] == "gray"
    assert captured["plot"][2][0] == [0.02, 0.03]
    assert captured["plot"][2][1] == [0.2, 0.1]
    assert captured["xlabel"] == "Physical error rate p"
    assert captured["ylabel"] == "Logical failure rate"
    assert captured["title"] == "Toric MWPM simulation"
    assert captured["grid"] == ((True,), {"alpha": 0.3})
    assert captured["legend"] == ((), {"loc": "upper left"})
    assert captured["savefig"][0] == figure_path
    assert captured["savefig"][1] == {"dpi": 200, "bbox_inches": "tight"}
    assert captured["closed"] is fake_fig
