import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

from toric_mwpm.simulation import SimulationResult, estimate_crossing, plot_results, run_point, run_sweep


def test_run_point_returns_valid_rate():
    result = run_point(L=4, p=0.05, trials=20, seed=7)

    assert result.L == 4
    assert result.p == 0.05
    assert result.trials == 20
    assert 0 <= result.failures <= 20
    assert 0.0 <= result.failure_rate <= 1.0
    assert result.standard_error >= 0.0


def test_run_sweep_returns_all_parameter_combinations():
    results = run_sweep(L_values=[4, 6], p_values=[0.05, 0.10], trials=5, seed=11)

    assert len(results) == 4
    assert {(r.L, r.p) for r in results} == {(4, 0.05), (4, 0.10), (6, 0.05), (6, 0.10)}


def test_estimate_crossing_uses_nearest_curve_difference():
    results = [
        SimulationResult(L=4, p=0.09, trials=100, failures=40),
        SimulationResult(L=8, p=0.09, trials=100, failures=30),
        SimulationResult(L=4, p=0.10, trials=100, failures=45),
        SimulationResult(L=8, p=0.10, trials=100, failures=46),
        SimulationResult(L=4, p=0.11, trials=100, failures=50),
        SimulationResult(L=8, p=0.11, trials=100, failures=60),
    ]

    assert estimate_crossing(results) == 0.10


def test_plot_results_closes_figure_if_savefig_fails(monkeypatch, tmp_path):
    results = [SimulationResult(L=4, p=0.1, trials=10, failures=1)]

    original_savefig = Figure.savefig

    def raise_savefig(self, *args, **kwargs):
        raise RuntimeError("save failed")

    monkeypatch.setattr(Figure, "savefig", raise_savefig)

    with pytest.raises(RuntimeError, match="save failed"):
        plot_results(results, tmp_path / "plot.png")

    assert plt.get_fignums() == []

    monkeypatch.setattr(Figure, "savefig", original_savefig)
