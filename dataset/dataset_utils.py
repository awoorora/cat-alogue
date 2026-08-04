"""
dataset_utils.py
===============
Shared primitives. build_dataset.py uses these to create the dataset,
and the baseline / model / evaluation code imports the SAME functions.
This file is the practical half of the "data contract". A shape or rule
changed here needs to be communicated to the whole team

Contents:
    resample_uniform(t, pos, dt)        linear resample onto a uniform time grid
    derive_velocity(pos, dt)            velocity via central differences
    label_regime(vel, dt, ...)          heuristics CV / CA / CT / MIX label
    add_noise(pos, sigma, seed=None)    add Gaussian measurement noise
    load_split(path)                    load a saved .npz split into a dict
"""

import numpy as np

# Regime label codes used everywhere (dataset field `regime`, evaluation breakdown).
#   CV = constant velocity          CA = (constant) acceleration / braking
#   CT = coordinated turn           MIX = accelerating through a turn / transition
REGIME_NAMES = {0: "CV", 1: "CA", 2: "CT", 3: "MIX"}

def resample_uniform(t, pos, dt):
    """
    Resample an irregularly-timed trajetory onto a uniform `dt` grid with LINEAR
    interpolation (as opposed to SPLINE).
    Why linear and not spline?: Linear interpolation stays exactly on the straight
    chord between two real samples, so it can never overshoot or invent motion. The
    raw samples are only ~7-10 ms apart, so that chord sits sub-millimetre from the
    true curved path, thousands of times smaller than the metre-scale noise added
    later. Splines can ring/overshoot and fabricate wiggles that were never flown,
    so we avoid them on purpose.

    :param t: (M,) timestamps in seconds, increasing, typically starting at 0
    :param pos: (M, 3) positions (x, y, z)
    :param dt: target timestep, e.g. 0.02s for 50 Hz
    :return: (grid_t (L,), pos_uniform (L, 3))
    """
    t = np.asarray(t, dtype=float)
    pos = np.asarray(pos, dtype=float)
    # Uniform grid spanning the segment
    grid_t = np.arange(t[0], t[-1], dt)
    pos_u = np.empty((len(grid_t), 3), dtype=float)
    for k in range(3):
        pos_u[:, k] = np.interp(grid_t, t, pos[:, k])
    return grid_t, pos_u

def derive_velocity(pos, dt):
    """
    Velocity from CLEAN positions using central differences (np.gradient):
        v[i] - (p[i+1] - p[i-1]) / (2*dt)
    Central differences are more accurate than forward differences and add no lag.
    Because `pos` here is the clean, resampled trajectory (noise is added later, not
    now), this velocity is ground truth

    :param pos: (L, 3) uniform-dt positions, given by the `resample_uniform` function
    :param dt: the constant sample spacing in seconds (e.g. 0.02s)
    :return: vel (L, 3)
    """
    return np.gradient(pos, dt, axis=0)

def _moving_average(a, k):
    """
    Computes a centered 1D moving average over a sequence using a uniform box filter.
    Smooths noise or high-frequency jitter in 1D arrays (such as velocity or acceleration
    signals used for regime classification)

    :param a: (N,) array-like, 1D input sequence to smooth
    :param k: int, filter window size in sampled. If k <= 1 returns original array
    :return: (N,) ndarray, smoothed array with the exact same length as `a`
    """
    if k <= 1:
        return a
    kernel = np.ones(k) / k
    # `mode="same"` convolution, meaning boundary samples (nead indices 0 and N-1) are
    # calculated with partial overlap, which slightly attenuates edges compared to full
    # window coverage
    return np.convolve(a, kernel, mode="same")

def label_regime(vel, dt, accel_thresh=6.0, turn_thresh=6.0, smooth=9):
    """
    Heuristics motion-regime label for a span of clean velocity.
    Method: split the acceleration into the component ALONG the velocity (tangential =
    speeding up / braking) and the component PERPENDICULAR to it (normal = turning),
    then classify by which is significant:
        low both                -> 0 CV     constant velocity
        high tangential only    -> 1 CA     accelerating / braking in a straight-ish line
        high normal only        -> 2 CT     coordinated turn (roughly constant speed)
        high both               -> 3 MIX    accelerating through a turn / regime transition
    IMPORTANT: real human flight blends regimes continuously, so these labels are
    something we IMPOSE, not ground truth. They are used only for the evaluation
    breakdown ("accuracy vs regime") and NEVER for training, so mild fuzziness can't
    hurt the model. Tune `accel_thresh` / `turn_thresh` (both in m/s^2) by coloring a
    trajectory by its label and nudging until the labels match what the path visibly
    does.

    :param vel: (L, 3) array-like, velocity vectors over time (in m/s)
    :param dt: float, uniform  sampling interval (in seconds)
    :param accel_thresh: float, acceleration threshold along track (m/s^2) to trigger CA/MIX
    :param turn_thresh: float, normal/perpendicular acceleration threshold (m/s^2) to trigger CT/MIX
    :param smooth: int, window size for moving average de-jittering prior to median reduction
    :return: int, scalar regime code (0: CV, 1: CA, 2: CT, 3: MIX)
    """
    vel = np.asarray(vel, dtype=float)

    if smooth > 1:
        vel_smooth = np.zeros_like(vel)
        for k in range(3):
            vel_smooth[:, k] = _moving_average(vel[:, k], smooth)
    else:
        vel_smooth = vel

    speed = np.linalg.norm(vel_smooth, axis=1)             #(L,)
    accel = np.gradient(vel, dt, axis=0)    #(L, 3) = d(vel)/dt

    # Unit tangent (direction of travel). Undefined when nearly stationary, so guard
    # the divide and treat hovering as "no maneuver" below
    eps = 1e-6
    tangent = vel / np.maximum(speed, eps)[:, None] # (L, 3)

    a_tan = np.sum(accel * tangent, axis=1)         # signed along-track accel (L,)
    a_perp_vec = accel - a_tan[:, None] * tangent   # sideways accel vector (L, 3)
    a_norm = np.linalg.norm(a_perp_vec, axis=1)     # turning accel magnitude (L,)

    # Where basically hovering, zero the maneuver signals (direction is meaningless).
    moving = speed > eps
    a_tan = np.where(moving, np.abs(a_tan), 0.0)
    a_norm = np.where(moving, np.abs(a_norm), 0.0)

    # De-jitter, then take a robust representative value over the whole span
    tan_rep = np.percentile(_moving_average(a_tan, smooth), 40)
    norm_rep = np.percentile(_moving_average(a_norm, smooth), 40)

    turning = norm_rep > turn_thresh
    accelerating = tan_rep > accel_thresh

    if turning and accelerating:
        return 3
    if turning:
        return 2
    if accelerating:
        return 1
    return 0

def add_noise(pos, sigma, seed=None):
    """
    Add isotropic Gaussian MEASUREMENT noise (std = sigma metred, independent per axis)
    to positions. This is how the model's noisy INPUT is produced from clean stored
    positions
    Call it fresh every training batch (seed=None) so the model sees new noise each time
    (free augmentation, and robustness to noise). Call it with a fixed `seed` during
    evaluation so a noise sweep is reproducible

    :param pos: (L, 3) array-like, clean (x, y, z) trajectory positions
    :param sigma: float, standard deviation of the Gaussian noise (in meters!)
    :param seed: int or None, seed for the random number generator (None for fresh noise)
    :return: (L, 3) ndarray, noisy position array matching the shape of `pos`
    """
    rng = np.random.default_rng(seed)
    return pos + rng.normal(0.0, sigma, size=np.shape(pos))

def load_split(path):
    """
    Load a split written by build.dataset.py into a plain dict
        input_pos   (N, W, 3)   clean input positions (call add_noise on these)
        future_pos  (N, H, 3)   clean future positions (prediction targets)
        state_gt    (N, W, 6)   clean x,y,z,vx,vy,vz over the input window
        regime      (N,)        regime labels per window (see REGIME_NAMES)
        seg_id      (N,)        source-segment index (debugging/traceability)
        dt          float       uniform timestep

    :param path: str or Path, path to the `.npz` binary file containing the dataset split
    :return: dict, dictionary mapping array names (strings) to NumPy arrays or scalars
    """
    d = np.load(path)
    out = {k: d[k] for k in d.files}
    out["dt"] = float(out["dt"]) # was stored as a 0-d array
    return out