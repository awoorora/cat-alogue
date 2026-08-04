# eval_harness

Adaptive trajectory prediction. It scores any predictor against ground truth across the three axes: measurement noise, prediction horizon, and motion regime; benchmarks a learned model against a fixed baseline, and produces the figures + summary used in the report.

Companion to the data pipeline (`cat-apult`) and the training repo (`cat_walk`).

---

## What it evaluates

Two predictors, run through identical code so their numbers are directly comparable:

- **Baseline (Kalman)** — a constant-velocity Kalman filter (`kalman_filter/catman.py`). The "simple fixed-model baseline" the challenge asks us to beat.
- **ML Model (LSTM)** — the trained sequence model `instinct_v2.keras`, wrapped in `machine_learning/predictor_lstm.py`.

Both expose the same two-method interface, so swapping models is a one-line change:

```python
predict(past_pos, dt, horizons)   # (W,3) noisy positions in -> (len(horizons),3) out
estimate_state(past_pos, dt)      # -> (pos, vel), each (3,)
```

Each also provides a batched `predict_batch` / `estimate_state_batch` that the harness uses automatically for speed (one model call for the whole test set instead of per-window).

---

## Metrics

The primary metric is **mean Euclidean position error in metres** — the standard trajectory-forecasting pair:

- **ADE** (Average Displacement Error): mean error over the whole predicted horizon and all windows — overall accuracy in one number.
- **FDE** (Final Displacement Error): error at the furthest horizon — how far the prediction has drifted by the end.

A velocity-estimation error (via `estimate_state`) is reported alongside, and a trajectory "corridor IoU" is kept only as a secondary sanity number.

---

## Running

From the repo root, with the venv active:

```bash
python eval_harness.py
```

It loads `dataset/test.npz`, runs all three sweeps for both predictors, prints a summary table, and writes three PNGs to `evaluation_plots/`. No arguments needed; the plots render headlessly (no GUI window required).

---

## Output

| File | Axis | What it shows |
| --- | --- | --- |
| `evaluation_plots/noise_sweep_analysis.png` | noise | ADE and velocity error vs measurement noise σ (0.5–8 m) |
| `evaluation_plots/horizon_degradation_sweep.png` | horizon | mean position error vs how far ahead we predict |
| `evaluation_plots/regime_performance_breakdown.png` | regime | ADE and velocity error broken down by motion regime |

Regimes: `CV` (constant velocity), `CA` (accelerating/braking), `CT` (coordinated turn), `MIX` (accelerating through a turn / transition).

**Reading the results:** the learned model's advantage is concentrated where the challenge cares — the harder regimes (`CT`, `MIX`) and robustness as noise grows — while the Kalman is near-optimal on simple constant-velocity motion (it *is* a constant-velocity model). See the report for the full interpretation.

> The LSTM was trained on measurement noise σ ∈ [0, 2] m. Sweep points at σ = 4 and 8 m are therefore **out of distribution** (extrapolation) and are shaded red on the noise plots. Read those points as robustness beyond spec, not in-distribution accuracy.

---

## Layout

```
eval_harness.py                     # the harness: sweeps, metrics, plots, summary
dataset/
    dataset_utils.py                # shared primitives (load_split, add_noise, REGIME_NAMES, ...)
    test.npz                        # held-out test split (windows + ground truth)
kalman_filter/
    catman.py                       # KalmanBaseline (the fixed-model baseline)
machine_learning/
    predictor_lstm.py               # Predictor wrapper around the Keras model
    instinct_v2.keras               # trained LSTM (BiLSTM, velocity preprocessing baked in)
    LSTM_on_npz.ipynb               # training/exploration notebook
evaluation_plots/                   # generated figures (regenerated on each run)
```

### Notes

- `catman.py` exports the aliases `Catman` / `Kitten` for `KalmanBaseline` — cosmetic, harmless.
- The dataset stores **clean** positions; the harness adds measurement noise at runtime with a fixed seed, so every predictor faces identical noise at each σ (fair comparison), and the noise sweep is trivial.
- The Kalman's measurement-noise `R` is set to the actual sweep σ at each point, so it is a properly-tuned (strong, honest) baseline rather than a mistuned one.
