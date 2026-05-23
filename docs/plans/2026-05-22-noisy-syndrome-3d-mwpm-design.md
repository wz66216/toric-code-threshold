# 3D Spacetime MWPM Extension Design

## Purpose

This extension adds noisy syndrome measurement to the existing two-dimensional toric-code MWPM simulator. The current simulator reproduces the zero-temperature/minimum-energy decoder result for independent bit-flip noise with perfect syndrome measurement, where the logical failure curves cross near the two-dimensional RBIM threshold scale \(p_c \approx 0.10\). The new extension targets the phenomenological noisy-measurement model: data qubits still live on the edges of an \(L\times L\) toric lattice, but each stabilizer measurement can be flipped by measurement noise. Repeated syndrome extraction creates a three-dimensional spacetime decoding problem.

The scientific goal is to demonstrate the qualitative threshold reduction predicted by Dennis--Kitaev--Landahl--Preskill and Wang--Harrington--Preskill: perfect syndrome decoding maps to a 2D random-bond Ising model, while noisy syndrome decoding maps to a 3D spacetime matching problem / random plaquette gauge model. The expected threshold scale drops from roughly \(10\%\) in the perfect-measurement setting to roughly \(3\%\) in the phenomenological noisy-measurement setting. The implementation should prioritize correctness, reproducibility, and report-ready evidence over high-precision finite-size scaling.

## Model

The spatial code remains the existing \(L\times L\) toric code representation with Boolean horizontal and vertical edge arrays. A noisy-memory trial runs for \(T\) rounds, usually with \(T=L\) by default.

For each time round:

1. Independent data errors are sampled on each spatial edge with probability \(p\).
2. The cumulative data error chain is updated.
3. The true vertex syndrome of the cumulative chain is computed.
4. Each measured syndrome bit is flipped independently with probability \(q\).

The default phenomenological comparison uses \(q=p\), but the API should keep \(p\) and \(q\) separate so the report can discuss unequal data and measurement error rates. The final time boundary should be closed by a perfect syndrome measurement after the last noisy round. This makes the detection-event history even and avoids hiding residual defects at the final time boundary.

Detection events are defined by the difference between consecutive measured syndrome layers:

\[
d_t = \tilde{s}_t \oplus \tilde{s}_{t-1},
\]

where \(\tilde{s}_t\) is the measured syndrome at time \(t\). The initial previous syndrome is the all-zero layer. For the final perfect boundary layer, the same difference rule is applied between the last noisy measurement and the final true syndrome.

## Decoder

The decoder constructs a complete matching graph over all detection events \((x,y,t)\). Space is periodic in \(x\) and \(y\); time is open. The distance between two events is an anisotropic spacetime Manhattan distance:

\[
D(a,b)=w_s\bigl(d_x^{\mathrm{toric}}+d_y^{\mathrm{toric}}\bigr)+w_t |t_a-t_b|.
\]

The default weights are negative-log-likelihood ratios:

\[
w_s = \log\frac{1-p}{p}, \qquad w_t = \log\frac{1-q}{q}.
\]

The complete-graph approach is not asymptotically optimal, but it is transparent and consistent with the existing 2D implementation. It is suitable for small and medium course-report sweeps. NetworkX MWPM is used to pair detection events. Each matched pair is then converted into a recovery contribution: the decoder follows a deterministic shortest spacetime path, accumulating only the spatial components into a final two-dimensional recovery chain. Time-like segments explain measurement-error pairings and do not directly add data recovery edges.

## Logical Failure Criterion

At the end of a trial, the simulator has:

- the total actual data-error chain \(E_{\mathrm{tot}}\), accumulated across all noisy rounds;
- the decoder's final spatial recovery chain \(R\), reconstructed from matched spacetime paths.

The residual chain is

\[
E_{\mathrm{tot}} + R.
\]

As in the existing 2D simulator, the run is a logical failure if this closed residual chain has nontrivial toric winding around either homology cycle. The fixed-cut convention from the 2D code is preserved, so the new result remains directly comparable with the existing perfect-syndrome threshold curves.

## Outputs

The extension should add a separate noisy-syndrome simulation layer and command-line script rather than overloading the existing perfect-measurement CLI. Expected outputs are:

- `outputs/noisy_threshold.csv` containing `L,T,p,q,trials,failures,failure_rate,standard_error`;
- `outputs/noisy_threshold.png` plotting logical failure rate versus \(p\) for several \(L\) values;
- optional side-by-side comparison guidance in the README explaining why the noisy threshold should be much lower than the perfect-measurement threshold.

The report interpretation should be cautious: small lattices and modest sample counts should be presented as qualitative evidence for the threshold shift, not as a precision estimate of \(p_c=0.0293\).

## Verification Strategy

The implementation must include tests for the spacetime logic before any threshold sweeps are trusted:

1. With \(p=q=0\), no detection events occur and the logical failure rate is zero.
2. A single measurement error at one vertex and one time produces two detection events at the same spatial coordinate in adjacent time layers.
3. A single data edge error produces spatially separated detection events consistent with the edge boundary.
4. The spacetime shortest-path helper respects periodic spatial distance and open time distance.
5. A small noisy-syndrome smoke sweep writes CSV and PNG outputs.
6. The existing 2D perfect-syndrome tests must continue to pass.

## Scope Boundaries

This design intentionally does not implement a full 3D topological code, 3D color code, X-cube code, or direct Monte Carlo simulation of the random plaquette gauge model. Those are valuable research directions but would require different stabilizer geometry, observables, and decoders. The chosen scope is the smallest 3D extension that is scientifically aligned with the paper: repeated syndrome measurement turns a 2D code into a 3D spacetime decoding problem.
