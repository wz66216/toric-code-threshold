import numpy as np

from toric_mwpm.decoder import decode_mwpm, toric_distance
from toric_mwpm.lattice import ToricChain, shortest_path_chain


def test_toric_distance_uses_periodic_shortcut():
    assert toric_distance((0, 0), (4, 0), 5) == 1
    assert toric_distance((0, 0), (3, 3), 5) == 4


def test_shortest_path_has_requested_boundary():
    path = shortest_path_chain(5, (4, 1), (1, 1))

    assert set(path.defects()) == {(4, 1), (1, 1)}
    assert path.weight() == 2


def test_decoder_recovers_single_edge_error_trivially():
    error = ToricChain.empty(5).with_horizontal(2, 3)
    recovery = decode_mwpm(error.syndrome())
    closed = error + recovery

    assert np.array_equal(recovery.syndrome(), error.syndrome())
    assert closed.syndrome().sum() == 0
    assert not closed.is_logical_failure()


def test_decoder_recovery_matches_random_syndrome_boundary():
    rng = np.random.default_rng(123)
    for _ in range(25):
        error = ToricChain(rng.random((6, 6)) < 0.08, rng.random((6, 6)) < 0.08)
        recovery = decode_mwpm(error.syndrome())
        assert np.array_equal(recovery.syndrome(), error.syndrome())
