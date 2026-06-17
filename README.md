# Toric Code MWPM Threshold Reproduction

This project numerically reproduces the two-dimensional toric-code minimum-weight decoding threshold under independent bit-flip noise and perfect syndrome measurement.

The simulation generates random error chains, computes the syndrome, decodes with minimum-weight perfect matching, and checks whether the combined error-plus-recovery chain is homologically nontrivial. The resulting logical-failure-rate curves cross near `p≈0.10`, consistent with the zero-temperature/minimum-energy RBIM threshold reported by Wang, Harrington, and Preskill.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run tests

```bash
PYTHONPATH=src python -m pytest -v
```

On Windows PowerShell, use:

```powershell
$env:PYTHONPATH="src"; python -m pytest -v
```

## Quick simulation

```bash
PYTHONPATH=src python scripts/run_threshold.py --sizes 6,8,10 --trials 100
```

On Windows PowerShell, use:

```powershell
$env:PYTHONPATH="src"; python scripts/run_threshold.py --sizes 6,8,10 --trials 100
```

Outputs are written to `outputs/toric_mwpm_threshold.csv` and `outputs/toric_mwpm_threshold.png` by default.

### Higher-quality threshold sweep

For the course report, use at least 2000–5000 trials and a finer p-grid near the expected crossing:

```powershell
$env:PYTHONPATH="src"; python scripts/run_threshold.py --sizes 6,8,10,12 --p-values 0.07,0.08,0.09,0.095,0.10,0.103,0.105,0.11,0.12,0.13 --trials 2000 --csv outputs/report_threshold.csv --figure outputs/report_threshold.png
```

The `toric_mwpm.simulation.estimate_crossing` function uses a simple heuristic (nearest-difference between smallest and largest L). For the report, interpret the crossing visually: where the curves for increasing L switch from "larger L → lower failure rate" to "larger L → higher failure rate".

## Final course-report analysis

Run the final noisy crossing sweep:

```powershell
python scripts/run_noisy_threshold.py --sizes 4,6,8 --p-values 0.024,0.026,0.028,0.030,0.032,0.034,0.036,0.038,0.040,0.042 --trials 1000 --seed 20260522 --csv outputs/noisy_crossing_fine.csv --figure outputs/noisy_crossing_fine.png
```

Run the fixed-size measurement sensitivity sweep:

```powershell
python scripts/run_measurement_sensitivity.py --L 6 --T 6 --p-values 0.020,0.025,0.030,0.035,0.040 --q-ratios 0.5,1.0,2.0 --trials 1000 --seed 20260522 --csv outputs/measurement_noise_sensitivity.csv
```

Run a decoder-mismatch scan, where the true measurement noise is `q=p` but the decoder assumes different `q_decode` values:

```powershell
python scripts/run_decoder_mismatch.py --L 6 --T 6 --p-values 0.025,0.030,0.035,0.040 --true-q-ratio 1.0 --decoder-q-ratios 0.5,1.0,2.0 --trials 1000 --seed 20260522 --csv outputs/decoder_mismatch.csv
```

Generate a 3D spacetime detection-event visualization for the noisy-syndrome discussion:

```powershell
python scripts/plot_spacetime_detection.py --L 6 --T 6 --p 0.03 --q 0.03 --seed 42 --figure outputs/report/spacetime_detection.png
```

Analyze all final report artifacts:

```powershell
python scripts/analyze_report_results.py --perfect-csv outputs/better_threshold.csv --noisy-csv outputs/noisy_crossing_fine.csv --sensitivity-csv outputs/measurement_noise_sensitivity.csv --mismatch-csv outputs/decoder_mismatch.csv --spacetime-figure outputs/report/spacetime_detection.png --out-dir outputs/report
```

Safe wording for the report: finite-size apparent crossing, not precision WHP reproduction; q-ratio sensitivity is fixed-size logical-failure sensitivity, not a threshold shift.

## Error correction visualization

Generate a 5-frame visualization of a single trial:

```powershell
$env:PYTHONPATH="src"; python -c "from toric_mwpm.animation import save_trial_frames; save_trial_frames(6, 0.10, 42, 'outputs/trial_viz')"
```

Frames show: (1) error chain, (2) syndrome defects, (3) MWPM pairing, (4) recovery chain, (5) combined chain with homology result. Pre-built demo frames are in `outputs/trial_demo/`, `outputs/trial_success/`, and `outputs/trial_failure/`.

## Report interpretation

MWPM chooses a minimum-weight recovery chain, so it corresponds to the zero-temperature/minimum-energy limit of the RBIM decoder. It should be compared with the reported `p_c0≈0.1031` value rather than the full Nishimori-line maximum-likelihood threshold near `p≈0.109`.

The simulation supports the course report's statistical-mechanical interpretation: local syndrome constraints define a boundary, while logical failure is determined by the homology class of the closed chain `E+R`.

## Noisy syndrome / 3D spacetime extension

This project also includes a phenomenological noisy syndrome model. Data qubit errors occur with probability `p`, measurement errors occur with probability `q`, and repeated syndrome extraction turns the problem into detection events in spacetime coordinates `(x, y, t)`.

Those detection events are decoded with 3D MWPM, where the time direction captures inconsistent syndrome records across rounds. By default, the noisy-syndrome sweep uses `q = p` and `T = L`.

Small smoke run:

```bash
python scripts/run_noisy_threshold.py --sizes 3,4 --p-values 0.02,0.03 --trials 5 --csv outputs/noisy_smoke.csv --figure outputs/noisy_smoke.png
```

Report-quality exploratory run:

```bash
python scripts/run_noisy_threshold.py --sizes 4,6,8 --p-values 0.015,0.02,0.025,0.03,0.035,0.04,0.05 --trials 500 --seed 20260522 --csv outputs/noisy_threshold.csv --figure outputs/noisy_threshold.png
```

Interpretation should stay cautious: repeated noisy measurement typically lowers the apparent threshold from the perfect-syndrome value near `p≈0.10` toward the noisy-measurement regime near `p≈0.03`, but this should not be overclaimed as a precision estimate versus the WHP reference `p_c0≈0.0293`.
