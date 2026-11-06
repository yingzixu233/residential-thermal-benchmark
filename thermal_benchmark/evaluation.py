"""Open-loop forecasts with recorded future forcing; no future temperatures."""
import numpy as np
from .data import FEATURES, TEMPERATURES


def metrics(pred, target):
    error = np.asarray(pred, dtype=float)-np.asarray(target, dtype=float)
    if not np.isfinite(error).all():
        raise FloatingPointError("Non-finite prediction: divergence is not silently clipped")
    return {"mae_C": float(np.mean(np.abs(error))),
            "rmse_C": float(np.sqrt(np.mean(error**2)))}


def temperature_noise(x, sigma, seed):
    out = np.asarray(x, dtype=float).copy()
    # Same perturbations across model families and budgets, paired by seed.
    out[:, :9] += np.random.default_rng(10000+seed).normal(0, sigma, (len(x), 9))
    return out


def recursive_rollout(model, frame, horizon=24, sigma=0., seed=0):
    if len(frame) <= horizon:
        raise ValueError("Not enough observations for a complete forecast")
    x = frame[FEATURES].to_numpy(dtype=float)
    y = frame[TEMPERATURES].to_numpy(dtype=float)
    origins = np.arange(len(frame)-horizon)
    state = temperature_noise(x[origins], sigma, seed)[:, :9]
    pred, target = [], []
    for h in range(horizon):
        forcing = x[origins+h, 9:]
        state = np.asarray(model.predict(np.column_stack([state, forcing])), dtype=float)
        if not np.isfinite(state).all():
            raise FloatingPointError(f"Non-finite rollout at horizon {h+1}")
        pred.append(state.copy())
        target.append(y[origins+h+1])
    return np.stack(pred, axis=1), np.stack(target, axis=1)
