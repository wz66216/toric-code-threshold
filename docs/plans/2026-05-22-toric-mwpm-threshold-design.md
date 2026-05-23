# Toric Code MWPM Threshold Reproduction Design

Date: 2026-05-22

## Goal

Reproduce a concrete quantum-error-correction threshold result from the course materials: the two-dimensional toric/surface-code threshold under independent bit-flip noise and perfect syndrome measurement using minimum-weight perfect matching (MWPM). The intended numerical result is a crossing of logical-failure-rate curves near

\[
p_{c0}\approx 0.10,
\]

consistent with the zero-temperature/minimum-energy threshold reported by Wang, Harrington, and Preskill for the 2D random-bond Ising model (RBIM), approximately \(p_{c0}=0.1031\). This deliberately reproduces the minimum-energy decoder threshold, not the full maximum-likelihood/Nishimori-line threshold near \(p\approx 0.109\). The distinction is useful for the final report because it numerically demonstrates why the minimum-energy threshold is below the Nishimori-line ML threshold.

## Scientific Context

For the toric code on an \(L\times L\) periodic square lattice, qubits live on edges. A bit-flip error configuration \(E\) is a set of errored edges. Its syndrome is the boundary \(\partial E\), i.e. the set of vertices incident to an odd number of errored edges. The syndrome does not uniquely identify the error chain; it only identifies its boundary. Any recovery chain \(R\) with \(\partial R=\partial E\) removes all local defects, but the combined closed chain \(E+R\) may contain a homologically nontrivial loop. Such a loop applies a logical operator and is counted as a decoding failure.

The statistical-mechanical mapping says that summing over all error chains in a stabilizer/logical equivalence class is a partition function of a quenched-disorder spin model. For the pure bit-flip toric-code case, this is the 2D RBIM. The Nishimori condition relates the physical error probability and inverse temperature by

\[
e^{-2\beta J}=\frac{p}{1-p}.
\]

Maximum-likelihood decoding corresponds to comparing partition functions of different logical classes on the Nishimori line. MWPM instead selects a minimum-weight chain and therefore corresponds to the zero-temperature/minimum-energy limit. The simulation will reproduce this minimum-energy threshold and use the report discussion to connect it to the broader Nishimori/RBIM picture.

## Chosen Numerical Reproduction

The implementation will simulate independent X errors on the periodic toric code and decode them with MWPM:

1. Generate Bernoulli error configurations on horizontal and vertical lattice edges.
2. Compute the vertex syndrome by parity checks.
3. Build a complete weighted graph on the defect vertices. Edge weights are shortest Manhattan distances on a torus.
4. Use MWPM to pair defects and construct a recovery path for each matched pair.
5. Add error and recovery chains mod 2.
6. Detect whether the resulting closed chain has nontrivial winding parity around either fundamental torus cycle.
7. Estimate the logical failure probability \(P_{\mathrm{fail}}(p,L)\) over many random trials.
8. Plot \(P_{\mathrm{fail}}\) versus \(p\) for several lattice sizes and estimate the crossing near \(p\approx 0.10\).

## Software Architecture

The project will be created from scratch because the folder contains papers and notes but no existing simulation code. The implementation will be a small Python package plus command-line scripts.

Planned files:

- `requirements.txt`: Python dependencies, expected to include `numpy`, `matplotlib`, `networkx`, and `pytest`.
- `src/toric_mwpm/__init__.py`: package export surface.
- `src/toric_mwpm/lattice.py`: toric lattice data representation, syndrome calculation, winding checks, and path construction.
- `src/toric_mwpm/decoder.py`: MWPM graph construction and recovery generation.
- `src/toric_mwpm/simulation.py`: Monte Carlo trial execution, parameter sweeps, aggregation, and optional bootstrap errors.
- `scripts/run_threshold.py`: command-line entry point for generating numerical results and figures.
- `tests/test_lattice.py`: deterministic tests for syndrome and homology logic.
- `tests/test_decoder.py`: decoder tests on simple known configurations.

The implementation will prefer clarity over maximum speed. The target course reproduction only needs modest lattice sizes and sample counts by default, with command-line options allowing larger runs.

## Data Model

An error chain is represented by two Boolean arrays:

- `horizontal[x, y]`: edge from vertex `(x, y)` to `(x+1 mod L, y)`.
- `vertical[x, y]`: edge from vertex `(x, y)` to `(x, y+1 mod L)`.

The syndrome array has shape `(L, L)`. A vertex is defective if the four incident edge parities sum to one mod 2.

For homology detection, the closed chain's winding parity is computed using fixed cuts:

- x-direction logical parity: horizontal edges crossing from `x=L-1` to `x=0`.
- y-direction logical parity: vertical edges crossing from `y=L-1` to `y=0`.

A logical failure occurs if either parity is odd.

## MWPM Details

For a set of syndrome defects, the decoder builds a complete graph whose nodes are defects. Each edge stores:

- toric Manhattan distance,
- one deterministic shortest path connecting the two defects,
- optional tie-breaking metadata.

`networkx.algorithms.matching.min_weight_matching` will compute a minimum-weight perfect matching. For each matched pair, the stored shortest path is added to the recovery chain mod 2. The resulting recovery must have the same syndrome as the original error, and tests will assert this invariant.

Periodic shortest paths can have ties. The design uses a deterministic tie-breaking rule for reproducibility. Ties may affect individual recoveries but should not bias threshold-scale results after random sampling.

## Experiments and Outputs

The default script should support a quick verification run and a heavier report-quality run.

Quick run target:

- lattice sizes: `L=6,8,10`,
- probabilities: a coarse grid around `0.07` to `0.13`,
- samples per point: small enough to finish quickly,
- purpose: verify the full pipeline and generate a preliminary plot.

Report run target:

- lattice sizes: e.g. `L=8,12,16,24`,
- probabilities: denser grid around `0.08` to `0.12`,
- samples per point: user-adjustable, typically hundreds to thousands,
- purpose: produce a clear logical-failure-rate crossing near `p≈0.10`.

Expected outputs:

- CSV file with columns such as `L`, `p`, `trials`, `failures`, `failure_rate`, and standard error.
- PNG/PDF plot of logical failure probability versus physical error probability.
- Console summary estimating the approximate crossing region.

## Testing and Verification

Before trusting the Monte Carlo output, the following checks are required:

1. No-error configuration has empty syndrome and no logical failure.
2. A single-edge error has two defects; after decoding, `error + recovery` is closed and logically trivial.
3. A manually constructed nontrivial loop has empty syndrome and logical failure.
4. Random decoder outputs satisfy `syndrome(recovery) == syndrome(error)`.
5. A small smoke simulation runs end-to-end and produces nonnegative failure rates in `[0,1]`.

The final verification will run the test suite and at least one simulation command that generates a figure.

## Report Integration

The final course report can use this simulation in the numerical-reproduction section as follows:

- derive toric-code syndrome decoding as chain homology;
- state the RBIM/Nishimori mapping;
- explain that MWPM is the \(T=0\) limit of the statistical-mechanical decoder;
- present the generated failure-rate crossing plot;
- compare the observed crossing near \(0.10\) with \(p_{c0}\approx0.1031\) and with the larger Nishimori/ML threshold around \(0.109\);
- discuss limitations: finite size, finite sample count, deterministic tie-breaking, and use of minimum-energy rather than full ML decoding.

## Out of Scope

The first implementation will not attempt full finite-temperature RBIM Monte Carlo, full ML decoding, 3D random plaquette gauge model simulation, color-code thresholds, or fracton-code thresholds. These are valuable extensions but would add substantial complexity and risk. A later extension may add small-lattice partition-function enumeration to demonstrate the Nishimori partition-function identity more directly.
