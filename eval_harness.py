"""
eval_harness.py  (Task D)
=========================
Evaluates any Contract-2 predictor (Kalman baseline, LSTM) against ground truth on the
test split, across the three required axes: noise, horizon, regime. Plus a state
(velocity) estimation check. Produces three figures + a printed summary.

PRIMARY METRIC = mean Euclidean position error, in metres (ADE / FDE):
    ADE (Average Displacement Error): mean error over the whole predicted horizon and
        all windows -- the single "how accurate overall" number.
    FDE (Final Displacement Error): error at the furthest horizon only -- how far the
        prediction has drifted by the end.
These are the standard trajectory-forecasting metrics and are what the challenge asks
for ("prediction accuracy against ground truth"). Unlike the corridor IoU we tried
first, ADE/FDE degrade smoothly and monotonically, so the plots stay legible at high
noise and long horizon. A trajectory-IoU "corridor hit-rate" is kept as a SECONDARY
sanity number only.

  !!! OUT-OF-DISTRIBUTION NOISE !!!
  The LSTM was trained on measurement noise sigma in [0, 2] m. Sweep points at
  sigma = 4 and 8 m are therefore OUTSIDE its training range -- the model is
  extrapolating there, so its numbers past 2 m must be read as "robustness beyond
  spec", not in-distribution accuracy. The noise plots shade the sigma > 2 m region to
  make this explicit. Either widen the LSTM's training noise to cover the sweep, or
  state this caveat in the report.

FAIRNESS NOTES:
  * Every predictor sees the IDENTICAL noisy input (fixed EVAL_SEED per sigma).
  * The Kalman baseline's measurement-noise R is set to the ACTUAL sweep sigma, so it
    is a properly-tuned (i.e. strong, honest) baseline rather than a mistuned one.
    Making the baseline strong is what makes the model's win credible.

Requires B's `predict_batch`/`estimate_state_batch` on KalmanBaseline and C's on
Predictor. Run from the eval_harness project root.
"""

import os
import numpy as np

import dataset.dataset_utils as dsu
import kalman_filter.catman as catman
import machine_learning.predictor_lstm as lstm

# Headless plotting: render straight to files, no GUI window (avoids the Tcl error).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("evaluation_plots", exist_ok=True)

# Regime code -> name, single source of truth (0=CV, 1=CA, 2=CT, 3=MIX)
REGIME_NAMES = dsu.REGIME_NAMES


# ----------------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------------
data = np.load("dataset/test.npz")
input_pos = data["input_pos"]      # (N, W, 3) clean input positions
future_pos = data["future_pos"]    # (N, H, 3) clean future positions (targets)
state_gt = data["state_gt"]        # (N, W, 6) clean x,y,z,vx,vy,vz over input window
regime = data["regime"]            # (N,) regime label per window
dt = float(data["dt"])             # scalar timestep (s)

N, W = input_pos.shape[:2]
H = future_pos.shape[1]
ALL_HORIZONS = list(range(1, H + 1))       # 1..H  (1-based steps; was range(40) = off-by-one)
HORIZON_SECONDS = np.array(ALL_HORIZONS) * dt

print(f"-> Samples N={N}  Window W={W}  Horizon H={H}  dt={dt}s ({1/dt:.0f} Hz)")


# ----------------------------------------------------------------------------------
# PREDICTOR ACCESS  (LSTM reused; Kalman rebuilt per sigma so its R matches the noise)
# ----------------------------------------------------------------------------------
lstm_model = lstm.Predictor("machine_learning/instinct_v2.keras")


def get_predictor(source, sigma):
    if source == "kalman":
        # tune the filter's measurement noise to the actual noise level (fair baseline)
        return catman.KalmanBaseline(sigma_meas=max(sigma, 1e-3))
    if source == "mlm":
        return lstm_model
    raise ValueError(source)


def get_predictions(source, sigma, seed):
    """(N, H, 3) predicted positions at every horizon 1..H, aligned with future_pos."""
    past_pos = dsu.add_noise(input_pos, sigma, seed)
    return get_predictor(source, sigma).predict_batch(past_pos, dt, ALL_HORIZONS)


def get_estimates(source, sigma, seed):
    """(N, 6) estimated [pos | vel] at the last input step."""
    past_pos = dsu.add_noise(input_pos, sigma, seed)
    pos, vel = get_predictor(source, sigma).estimate_state_batch(past_pos, dt)
    return np.concatenate([pos, vel], axis=1)


# ----------------------------------------------------------------------------------
# METRICS
# ----------------------------------------------------------------------------------
def displacement_error(pred, gt):
    """Per-window, per-horizon Euclidean error -> (N, H), in metres."""
    return np.linalg.norm(pred - gt, axis=-1)


def ade(err):   # mean over all windows and horizons
    return float(err.mean())


def fde(err):   # error at the final horizon
    return float(err[:, -1].mean())


def estimation_errors(est_state):
    """Position/velocity estimation error at the last input step -> (pos_err, vel_err)."""
    gt_pos = input_pos[:, -1, :]        # clean position at last input step
    gt_vel = state_gt[:, -1, 3:]        # clean velocity at last input step
    pos_err = np.linalg.norm(est_state[:, :3] - gt_pos, axis=1)
    vel_err = np.linalg.norm(est_state[:, 3:] - gt_vel, axis=1)
    return pos_err, vel_err


def regime_means(per_window_values):
    """Mean of a per-window quantity within each regime; nan where a regime is absent."""
    out = {}
    for code, name in REGIME_NAMES.items():
        mask = regime == code
        out[name] = float(per_window_values[mask].mean()) if mask.any() else np.nan
    return out


# --- SECONDARY corridor metric (kept from D's original, demoted to a sanity number) ---
def densify(coords, points_per_meter=5):
    if len(coords) == 1:
        return coords.copy()
    segs = []
    for i in range(len(coords) - 1):
        d = np.linalg.norm(coords[i + 1] - coords[i])
        segs.append(np.linspace(coords[i], coords[i + 1],
                                num=max(2, int(d * points_per_meter)), endpoint=False))
    segs.append(coords[-1:])
    return np.vstack(segs)


def trajectory_iou(pred_pos, future_pos, buffer_radius=0.5, points_per_meter=5):
    """Fraction of the predicted path within `buffer_radius` of the true path (0..1)."""
    scores = np.zeros(len(pred_pos))
    for n in range(len(pred_pos)):
        p = densify(pred_pos[n], points_per_meter)
        g = densify(future_pos[n], points_per_meter)
        d = np.linalg.norm(p[:, None, :] - g[None, :, :], axis=2)
        inter = np.sum(np.any(d < buffer_radius, axis=1))
        union = len(p) + len(g) - inter
        scores[n] = inter / union if union else 0.0
    return scores


# ----------------------------------------------------------------------------------
# SWEEPS
# ----------------------------------------------------------------------------------
print("\n=== Evaluation sweeps ===")
EVAL_SEED = 42            # fixed -> every predictor faces identical noise (fair)
NOMINAL_SIGMA = 0.5      # in-distribution reference noise for horizon & regime sweeps
SIGMAS = [0.5, 1.0, 2.0, 4.0, 8.0]
OOD_SIGMA = 2.0          # LSTM trained on [0, 2] m; beyond this it extrapolates

# --- Sweep 1: NOISE (ADE, FDE, velocity error at each sigma) ---
print("-> noise sweep")
sweep = {s: {} for s in ("kalman", "mlm")}
for src in ("kalman", "mlm"):
    ades, fdes, verrs, ious = [], [], [], []
    for sigma in SIGMAS:
        err = displacement_error(get_predictions(src, sigma, EVAL_SEED), future_pos)
        _, vel_err = estimation_errors(get_estimates(src, sigma, EVAL_SEED))
        ades.append(ade(err)); fdes.append(fde(err)); verrs.append(vel_err.mean())
    sweep[src] = dict(ade=ades, fde=fdes, vel=verrs)

# --- Sweeps 2 & 3: HORIZON and REGIME at the nominal (in-distribution) noise level ---
print("-> horizon + regime sweep")
nom = {}
for src in ("kalman", "mlm"):
    preds = get_predictions(src, NOMINAL_SIGMA, EVAL_SEED)
    err = displacement_error(preds, future_pos)          # (N, H)
    _, vel_err = estimation_errors(get_estimates(src, NOMINAL_SIGMA, EVAL_SEED))
    nom[src] = dict(
        per_horizon=err.mean(axis=0),                    # (H,) ADE at each horizon
        per_window=err.mean(axis=1),                     # (N,) ADE per window
        vel_err=vel_err,
        iou=trajectory_iou(preds, future_pos),           # secondary
    )


# ----------------------------------------------------------------------------------
# PLOTS
# ----------------------------------------------------------------------------------
print("\n=== Charts ===")
STYLE = {"kalman": dict(color="gray", ls="--", marker="o", label="Baseline (Kalman)"),
         "mlm":    dict(color="blue", ls="-",  marker="s", label="ML Model (LSTM)")}


def shade_ood(ax):
    """Mark the sigma > 2 m region where the LSTM is extrapolating."""
    ax.axvspan(OOD_SIGMA, max(SIGMAS), color="red", alpha=0.07)
    ax.text(OOD_SIGMA * 1.05, ax.get_ylim()[1] * 0.92, "LSTM out-of-distribution\n(σ > 2 m)",
            fontsize=8, color="firebrick", va="top")


# --- PLOT 1: noise sweep (ADE and velocity error) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for src in ("kalman", "mlm"):
    ax1.plot(SIGMAS, sweep[src]["ade"], **STYLE[src])
ax1.set_title("Position accuracy vs noise")
ax1.set_xlabel(r"measurement noise $\sigma$ (m)")
ax1.set_ylabel("ADE (m) — lower is better")
ax1.grid(True, ls=":", alpha=0.6); ax1.legend(); shade_ood(ax1)

for src in ("kalman", "mlm"):
    ax2.plot(SIGMAS, sweep[src]["vel"], **STYLE[src])
ax2.set_title("Velocity estimation error vs noise")
ax2.set_xlabel(r"measurement noise $\sigma$ (m)")
ax2.set_ylabel("velocity error (m/s) — lower is better")
ax2.grid(True, ls=":", alpha=0.6); ax2.legend(); shade_ood(ax2)
fig.tight_layout(); fig.savefig("evaluation_plots/noise_sweep_analysis.png", dpi=200)
plt.close(fig)
print("-> evaluation_plots/noise_sweep_analysis.png")

# --- PLOT 2: horizon degradation (ADE per horizon; now correctly increasing) ---
fig, ax = plt.subplots(figsize=(8, 5))
for src in ("kalman", "mlm"):
    ax.plot(HORIZON_SECONDS, nom[src]["per_horizon"], **STYLE[src])
ax.set_title(f"Position error vs prediction horizon (σ = {NOMINAL_SIGMA} m)")
ax.set_xlabel("prediction horizon (s)")
ax.set_ylabel("mean position error (m) — lower is better")
ax.grid(True, ls=":", alpha=0.6); ax.legend()
fig.tight_layout(); fig.savefig("evaluation_plots/horizon_degradation_sweep.png", dpi=200)
plt.close(fig)
print("-> evaluation_plots/horizon_degradation_sweep.png")

# --- PLOT 3: per-regime ADE and velocity error (all four regimes) ---
names = list(REGIME_NAMES.values())
xs = np.arange(len(names))
width = 0.38
k_ade = [regime_means(nom["kalman"]["per_window"])[n] for n in names]
m_ade = [regime_means(nom["mlm"]["per_window"])[n] for n in names]
k_vel = [regime_means(nom["kalman"]["vel_err"])[n] for n in names]
m_vel = [regime_means(nom["mlm"]["vel_err"])[n] for n in names]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.bar(xs - width/2, np.nan_to_num(k_ade), width, color="silver", label="Baseline (Kalman)")
ax1.bar(xs + width/2, np.nan_to_num(m_ade), width, color="royalblue", label="ML Model (LSTM)")
ax1.set_title(f"Position error by regime (σ = {NOMINAL_SIGMA} m)")
ax1.set_xticks(xs); ax1.set_xticklabels(names)
ax1.set_ylabel("ADE (m) — lower is better"); ax1.legend()
ax1.grid(True, axis="y", ls=":", alpha=0.6)

ax2.bar(xs - width/2, np.nan_to_num(k_vel), width, color="silver", label="Baseline (Kalman)")
ax2.bar(xs + width/2, np.nan_to_num(m_vel), width, color="royalblue", label="ML Model (LSTM)")
ax2.set_title("Velocity error by regime")
ax2.set_xticks(xs); ax2.set_xticklabels(names)
ax2.set_ylabel("velocity error (m/s) — lower is better"); ax2.legend()
ax2.grid(True, axis="y", ls=":", alpha=0.6)
fig.tight_layout(); fig.savefig("evaluation_plots/regime_performance_breakdown.png", dpi=200)
plt.close(fig)
print("-> evaluation_plots/regime_performance_breakdown.png")


# ----------------------------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------------------------
i_nom = SIGMAS.index(NOMINAL_SIGMA) if NOMINAL_SIGMA in SIGMAS else 0
print("\n" + "=" * 64)
print(f"SUMMARY at nominal sigma = {NOMINAL_SIGMA} m   (lower is better, except IoU)")
print("-" * 64)
print(f"{'metric':<22}{'Kalman':>12}{'LSTM':>12}")
kn, mn = nom["kalman"], nom["mlm"]
print(f"{'ADE (m)':<22}{kn['per_window'].mean():>12.3f}{mn['per_window'].mean():>12.3f}")
print(f"{'FDE (m)':<22}{kn['per_horizon'][-1]:>12.3f}{mn['per_horizon'][-1]:>12.3f}")
print(f"{'velocity err (m/s)':<22}{kn['vel_err'].mean():>12.3f}{mn['vel_err'].mean():>12.3f}")
print(f"{'trajectory IoU (2nd)':<22}{kn['iou'].mean():>12.3f}{mn['iou'].mean():>12.3f}")
print("-" * 64)
print("per-regime ADE (m):")
kr, mr = regime_means(kn["per_window"]), regime_means(mn["per_window"])
counts = {REGIME_NAMES[c]: int((regime == c).sum()) for c in REGIME_NAMES}
for name in names:
    kv = f"{kr[name]:.3f}" if not np.isnan(kr[name]) else "  n/a"
    mv = f"{mr[name]:.3f}" if not np.isnan(mr[name]) else "  n/a"
    print(f"  {name:<6} (n={counts[name]:>5}) {'Kalman':>8} {kv:>8} | {'LSTM':>5} {mv:>8}")
print("-" * 64)
print("NOTE: LSTM trained on sigma in [0, 2] m; sweep points at 4 and 8 m are")
print("      out-of-distribution (extrapolation), shaded red in the noise plot.")
print("=" * 64)
