import numpy as np
import csv

from toric_mwpm.lattice import chain_from_edges
from toric_mwpm.decoder import decode_mwpm
from toric_mwpm.noisy import (
    DetectionEvent,
    NoisySimulationResult,
    detection_events_from_measurements,
    measured_syndrome_history,
    decode_spacetime_mwpm,
    spacetime_distance,
    run_noisy_point,
    run_noisy_sweep,
    plot_noisy_results,
    write_noisy_csv,
)


def test_single_measurement_error_creates_adjacent_time_events():
    L = 3
    measurement_flips = [
        np.zeros((L, L), dtype=bool),
        np.zeros((L, L), dtype=bool),
    ]
    measurement_flips[0][1, 2] = True
    true_syndromes = [
        np.zeros((L, L), dtype=bool),
        np.zeros((L, L), dtype=bool),
        np.zeros((L, L), dtype=bool),
    ]

    measured = measured_syndrome_history(true_syndromes, measurement_flips)
    events = detection_events_from_measurements(measured)

    assert events == [DetectionEvent(1, 2, 0), DetectionEvent(1, 2, 1)]


def test_single_data_error_creates_spatial_detection_events():
    L = 4
    error = chain_from_edges(L, horizontal_edges=((1, 2),))
    true_syndromes = [
        np.zeros((L, L), dtype=bool),
        error.syndrome(),
    ]
    measurement_flips = [np.zeros((L, L), dtype=bool)]

    measured = measured_syndrome_history(true_syndromes, measurement_flips)
    events = detection_events_from_measurements(measured)

    assert events == [DetectionEvent(1, 2, 0), DetectionEvent(2, 2, 0)]


def test_spacetime_distance_uses_toric_space_and_open_time():
    a = DetectionEvent(0, 0, 0)
    b = DetectionEvent(4, 0, 0)
    c = DetectionEvent(0, 0, 3)

    assert spacetime_distance(a, b, L=5, space_weight=2.0, time_weight=7.0) == 2.0
    assert spacetime_distance(a, c, L=5, space_weight=2.0, time_weight=7.0) == 21.0


def test_decode_spacetime_mwpm_empty_events_returns_empty_chain():
    recovery = decode_spacetime_mwpm([], L=4, p=0.03, q=0.03)

    assert recovery.weight() == 0
    assert recovery.logical_parity() == (False, False)


def test_decode_spacetime_mwpm_with_single_round_data_error_matches_static_decoder():
    L = 4
    error = chain_from_edges(L, horizontal_edges=((1, 2),))
    events = [DetectionEvent(1, 2, 0), DetectionEvent(2, 2, 0)]

    recovery = decode_spacetime_mwpm(events, L=L, p=0.03, q=0.0)

    assert (error + recovery).syndrome().sum() == 0
    assert recovery.weight() == decode_mwpm(error.syndrome()).weight()


def test_decode_spacetime_mwpm_rejects_temporal_separation_when_q_zero():
    events = [DetectionEvent(0, 0, 0), DetectionEvent(0, 0, 1)]

    with np.testing.assert_raises_regex(ValueError, "finite perfect matching"):
        decode_spacetime_mwpm(events, L=4, p=0.03, q=0.0)


def test_decode_spacetime_mwpm_rejects_spatial_separation_when_p_zero():
    events = [DetectionEvent(0, 0, 0), DetectionEvent(1, 0, 0)]

    with np.testing.assert_raises_regex(ValueError, "finite perfect matching"):
        decode_spacetime_mwpm(events, L=4, p=0.0, q=0.03)


def test_decode_spacetime_mwpm_allows_same_location_temporal_pair_when_p_zero():
    events = [DetectionEvent(2, 1, 0), DetectionEvent(2, 1, 1)]

    recovery = decode_spacetime_mwpm(events, L=4, p=0.0, q=0.03)

    assert recovery.weight() == 0
    assert recovery.logical_parity() == (False, False)


def test_decode_spacetime_mwpm_allows_same_time_spatial_pair_when_q_zero():
    events = [DetectionEvent(0, 0, 0), DetectionEvent(1, 0, 0)]

    recovery = decode_spacetime_mwpm(events, L=4, p=0.03, q=0.0)

    assert recovery.weight() == 1


def test_decode_spacetime_mwpm_projected_recovery_closes_parity_for_mixed_separation():
    events = [DetectionEvent(0, 0, 0), DetectionEvent(2, 0, 1)]

    recovery = decode_spacetime_mwpm(events, L=4, p=0.03, q=0.03)

    assert recovery.syndrome().sum() == 2


def test_zero_noise_point_has_no_failures():
    result = run_noisy_point(L=4, T=4, p=0.0, q=0.0, trials=5, seed=123)

    assert result == NoisySimulationResult(L=4, T=4, p=0.0, q=0.0, trials=5, failures=0)
    assert result.failure_rate == 0.0
    assert result.standard_error == 0.0


def test_noisy_sweep_is_reproducible_with_seed():
    first = run_noisy_sweep(L_values=[3, 4], p_values=[0.02, 0.03], trials=3, seed=7)
    second = run_noisy_sweep(L_values=[3, 4], p_values=[0.02, 0.03], trials=3, seed=7)

    assert first == second
    assert [item.T for item in first] == [3, 3, 4, 4]


def test_write_noisy_csv_schema(tmp_path):
    results = [NoisySimulationResult(L=4, T=4, p=0.02, q=0.02, trials=10, failures=1)]
    output = tmp_path / "noisy.csv"
    write_noisy_csv(results, output)

    with output.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["L", "T", "p", "q", "trials", "failures", "failure_rate", "standard_error"]
        rows = list(reader)

    assert rows[0]["L"] == "4"
    assert rows[0]["T"] == "4"
    assert rows[0]["p"] == "0.02"
    assert rows[0]["q"] == "0.02"
    assert rows[0]["trials"] == "10"
    assert rows[0]["failures"] == "1"
    assert float(rows[0]["failure_rate"]) == 0.1
    assert float(rows[0]["standard_error"]) == float((0.1 * 0.9 / 10) ** 0.5)


def test_plot_noisy_results_groups_by_l_and_t(tmp_path):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    results = [
        NoisySimulationResult(L=3, T=3, p=0.02, q=0.02, trials=10, failures=1),
        NoisySimulationResult(L=3, T=5, p=0.03, q=0.03, trials=10, failures=2),
        NoisySimulationResult(L=4, T=4, p=0.04, q=0.04, trials=10, failures=0),
    ]
    output = tmp_path / "noisy.png"
    plot_noisy_results(results, output)

    assert output.exists()
    assert plt.get_fignums() == []


def test_plot_noisy_results_writes_file_and_closes_figures(tmp_path):
    import matplotlib.pyplot as plt

    results = [
        NoisySimulationResult(L=3, T=3, p=0.02, q=0.02, trials=10, failures=0),
        NoisySimulationResult(L=3, T=3, p=0.03, q=0.03, trials=10, failures=1),
    ]
    output = tmp_path / "noisy.png"
    plot_noisy_results(results, output)

    assert output.exists()
    assert plt.get_fignums() == []


def test_plot_noisy_results_writes_separate_series_for_same_l_different_t(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt

    captured = []

    import matplotlib.axes

    original_errorbar = matplotlib.axes.Axes.errorbar

    def capture_errorbar(self, *args, **kwargs):
        captured.append(kwargs.get("label"))
        return original_errorbar(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", capture_errorbar)

    results = [
        NoisySimulationResult(L=3, T=3, p=0.02, q=0.02, trials=10, failures=1),
        NoisySimulationResult(L=3, T=5, p=0.03, q=0.03, trials=10, failures=2),
    ]
    output = tmp_path / "noisy.png"

    plot_noisy_results(results, output)

    assert sorted(captured) == ["L=3, T=3", "L=3, T=5"]
    assert output.exists()
    assert plt.get_fignums() == []
