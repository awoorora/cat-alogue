import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.models import load_model


@keras.saving.register_keras_serializable()
class VelNorm(layers.Layer):
    """Baked-in preprocessing: (W,3) anchor-relative positions -> standardized velocities (W-1,3)."""

    def __init__(self, x_mean, x_std, **kw):
        super().__init__(**kw)
        self.x_mean = np.asarray(x_mean, np.float32)
        self.x_std = np.asarray(x_std, np.float32)

    def call(self, x):                               # x: (B, W, 3)
        v = x[:, 1:, :] - x[:, :-1, :]               # velocities (B, W-1, 3)
        return (v - self.x_mean) / self.x_std

    def get_config(self):
        c = super().get_config()
        c.update(x_mean=self.x_mean.tolist(), x_std=self.x_std.tolist())
        return c


@keras.saving.register_keras_serializable()
class Integrate(layers.Layer):
    """Cumulative sum over the horizon axis: per-step velocities -> offsets."""

    def call(self, z):
        return tf.cumsum(z, axis=1)


@keras.saving.register_keras_serializable()
def weighted_mse(y_true, y_pred):
    # Only needed to satisfy deserialization of the saved compile_config;
    # eval_harness only calls predict(), so the real weights/closure don't matter.
    return tf.reduce_mean(tf.square(y_pred - y_true))


class Predictor:
    def __init__(self, path):
        self.model = load_model(path, compile=False)

    def predict(self, past_pos, dt, horizons):
        past_pos = np.asarray(past_pos, dtype=np.float32)
        anchor = past_pos[-1]
        disp = self.model.predict((past_pos - anchor)[None], verbose=0)[0]
        future = disp + anchor
        return future[[h - 1 for h in horizons]]

    def predict_batch(self, past_pos, dt, horizons):
        """Batched predict. past_pos: (N, W, 3) -> (N, len(horizons), 3).
        One Keras call for the whole batch instead of N calls."""
        past = np.asarray(past_pos, dtype=np.float32)      # (N, W, 3)
        anchor = past[:, -1:, :]                            # (N, 1, 3)
        disp = self.model.predict(past - anchor, verbose=0)  # (N, H*3)
        disp = disp.reshape(len(past), -1, 3)               # (N, H, 3)
        future = disp + anchor                              # re-anchor (broadcasts)
        return future[:, [h - 1 for h in horizons], :]

    def estimate_state(self, past_pos, dt, k=20):
        seg = np.asarray(past_pos, dtype=np.float64)[-k:]
        t = np.arange(len(seg), dtype=np.float64)
        tm, pm = t.mean(), seg.mean(axis=0)
        slope = ((t - tm)[:, None] * (seg - pm)).sum(axis=0) / ((t - tm) ** 2).sum()
        return pm + slope * (t[-1] - tm), slope / dt

    def estimate_state_batch(self, past_pos, dt, k=20):
        """Batched (position, velocity) at the last fix. past_pos: (N, W, 3)
        -> (pos (N,3), vel (N,3)). Same least-squares line fit as estimate_state."""
        seg = np.asarray(past_pos, dtype=np.float64)[:, -k:, :]   # (N, k, 3)
        w = seg.shape[1]
        t = np.arange(w, dtype=np.float64)
        tm = t.mean()
        pm = seg.mean(axis=1)                                     # (N, 3)
        tc = t - tm
        slope = (tc[None, :, None] * (seg - pm[:, None, :])).sum(axis=1) / (tc ** 2).sum()
        pos = pm + slope * (t[-1] - tm)
        return pos, slope / dt
