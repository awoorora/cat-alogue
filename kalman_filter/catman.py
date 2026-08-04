# https://www.geeksforgeeks.org/python/kalman-filter-in-python/

from __future__ import annotations

import numpy as np

class KalmanBaseline:
    def __init__(
        self,
        sigma_meas: float = 1.0,  # measurement noise std 
        sigma_acc: float = 3.0,   # process noise: how much accel the CV model tolerates
    ):
        self.sigma_meas = float(sigma_meas)
        self.sigma_acc = float(sigma_acc)

        # --- Measurement H: we observe position only (independent of dt) ---
        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0

        # --- Measurement noise R (independent of dt) ---
        self.R = np.eye(3) * (self.sigma_meas ** 2)

        # dt-dependent matrices (F, Q) are built per call and cached here
        self._dt: float | None = None
        self.F = np.eye(6)
        self.Q = np.zeros((6, 6))

    def _build(self, dt: float) -> None:
        if self._dt == dt:
            return
        self._dt = dt

        # Transition F: position integrates velocity over dt
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Process noise Q: white-noise-acceleration, per axis (2x2) tiled into 6x6
        q = self.sigma_acc ** 2
        dt2, dt3, dt4 = dt ** 2, dt ** 3, dt ** 4
        block = np.array([[dt4 / 4.0, dt3 / 2.0],
                          [dt3 / 2.0, dt2]])
        self.Q = np.zeros((6, 6))
        for i in range(3):                       # x, y, z axes
            p, v = i, i + 3                       # position index, velocity index
            self.Q[p, p] = block[0, 0] * q
            self.Q[p, v] = block[0, 1] * q
            self.Q[v, p] = block[1, 0] * q
            self.Q[v, v] = block[1, 1] * q

    def _run(self, past_pos, dt: float) -> tuple[np.ndarray, np.ndarray]:
        self._build(dt)
        w = np.asarray(past_pos, dtype=float).reshape(-1, 3)
        if len(w) == 0:
            raise ValueError("empty window")

        # Initialise: position = first fix, velocity = finite difference if we can
        x = np.zeros(6)
        x[:3] = w[0]
        if len(w) >= 2:
            x[3:] = (w[1] - w[0]) / dt
        # Loose prior -> trust incoming measurements early
        P = np.eye(6) * 500.0

        for z in w[1:]:
            # predict
            x = self.F @ x
            P = self.F @ P @ self.F.T + self.Q
            # update
            y = z - self.H @ x                       # innovation
            S = self.H @ P @ self.H.T + self.R
            K = P @ self.H.T @ np.linalg.inv(S)      # Kalman gain
            x = x + K @ y
            P = (np.eye(6) - K @ self.H) @ P

        return x, P

    def predict(self, past_pos, dt, horizons):
        """Absolute future positions at the requested horizons."""

        x, _ = self._run(past_pos, dt)
        steps_max = max(horizons)
        traj = np.empty((steps_max, 3))
        for i in range(steps_max):
            x = self.F @ x                            # propagate
            traj[i] = x[:3]
        return traj[[h - 1 for h in horizons]]

    def predict_batch(self, past_pos, dt, horizons):
        """Vectorised predict over a whole batch. past_pos: (N, W, 3) NOISY windows.
        Returns (N, len(horizons), 3), numerically identical to looping predict()."""
        self._build(dt)
        W = np.asarray(past_pos, dtype=float)  # (N, W, 3)
        n, w, _ = W.shape

        x = np.zeros((n, 6))  # per-window state
        x[:, :3] = W[:, 0, :]
        if w >= 2:
            x[:, 3:] = (W[:, 1, :] - W[:, 0, :]) / dt
        P = np.broadcast_to(np.eye(6) * 500.0, (n, 6, 6)).copy()

        F, Q, H, R, I = self.F, self.Q, self.H, self.R, np.eye(6)
        for t in range(1, w):  # one loop over W, not N*W
            z = W[:, t, :]
            x = x @ F.T
            P = F @ P @ F.T + Q
            y = z - x @ H.T
            S = H @ P @ H.T + R  # (N,3,3)
            K = np.transpose(np.linalg.solve(S, np.transpose(P @ H.T, (0, 2, 1))), (0, 2, 1))
            x = x + np.einsum('nij,nj->ni', K, y)
            P = (I - K @ H) @ P

        steps = max(horizons)
        out = np.empty((n, steps, 3))
        for i in range(steps):
            x = x @ F.T
            out[:, i, :] = x[:, :3]
        return out[:, [h - 1 for h in horizons], :]

    def estimate_state(self, past_pos, dt):
        """Denoised (position, velocity) at the last fix."""
        k=20
        seg = np.asarray(past_pos, dtype=float).reshape(-1, 3)[-k:]
        x, _ = self._run(seg, dt)
        return x[:3].copy(), x[3:].copy()

    def estimate_state_batch(self, past_pos, dt, k=20):
        """Batched (position, velocity) at the last fix for every window.
        past_pos: (N, W, 3) NOISY. Returns (pos (N,3), vel (N,3))."""
        self._build(dt)
        W = np.asarray(past_pos, dtype=float)[:, -k:, :]  # (N, k, 3)
        n, w, _ = W.shape
        x = np.zeros((n, 6))
        x[:, :3] = W[:, 0, :]
        if w >= 2:
            x[:, 3:] = (W[:, 1, :] - W[:, 0, :]) / dt
        P = np.broadcast_to(np.eye(6) * 500.0, (n, 6, 6)).copy()
        F, Q, H, R, I = self.F, self.Q, self.H, self.R, np.eye(6)
        for t in range(1, w):
            z = W[:, t, :]
            x = x @ F.T
            P = F @ P @ F.T + Q
            y = z - x @ H.T
            S = H @ P @ H.T + R
            K = np.transpose(np.linalg.solve(S, np.transpose(P @ H.T, (0, 2, 1))), (0, 2, 1))
            x = x + np.einsum('nij,nj->ni', K, y)
            P = (I - K @ H) @ P
        return x[:, :3].copy(), x[:, 3:].copy()



Catman = KalmanBaseline
Kitten = KalmanBaseline
