import numpy as np

from toric_mwpm.lattice import ToricChain, chain_from_edges


def test_empty_chain_has_expected_shape_and_no_edges():
    chain = ToricChain.empty(4)

    assert chain.L == 4
    assert chain.horizontal.shape == (4, 4)
    assert chain.vertical.shape == (4, 4)
    assert chain.weight() == 0


def test_single_horizontal_edge_has_two_syndrome_defects():
    chain = ToricChain.empty(5).with_horizontal(1, 2)

    defects = set(chain.defects())

    assert defects == {(1, 2), (2, 2)}
    assert chain.syndrome().sum() == 2


def test_nontrivial_horizontal_loop_has_empty_syndrome_and_x_winding():
    chain = ToricChain.empty(5)
    for x in range(5):
        chain = chain.with_horizontal(x, 0)

    assert chain.syndrome().sum() == 0
    assert chain.logical_parity() == (True, False)
    assert chain.is_logical_failure()


def test_mod2_addition_cancels_identical_chains():
    chain = ToricChain.empty(4).with_vertical(3, 1)

    combined = chain + chain
    assert combined.weight() == 0
    assert not np.any(combined.syndrome())
