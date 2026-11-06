"""Strict chronological splits; no transition crosses a split boundary."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

TEMPERATURES = [f"T{i:02d}_TEMP" for i in range(1, 10)]
HEATING = [f"T{i:02d}_Wh" for i in range(1, 10)]
FEATURES = TEMPERATURES + HEATING + ["Text", "GHI"]
TRAIN_END, VALID_END, EXPECTED_ROWS = 240, 288, 385


def load_data(path):
    df = pd.read_csv(path, parse_dates=["Time"])
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} hourly observations, got {len(df)}")
    if not df.Time.is_monotonic_increasing or df.Time.duplicated().any():
        raise ValueError("Timestamps must be strictly increasing")
    if not df.Time.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Data must have exactly one-hour spacing")
    if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Missing or non-finite values are not silently imputed")
    return df


def transitions(df, start, stop):
    """Observations [start, stop) yield stop-start-1 supervised pairs."""
    return (df.iloc[start:stop-1][FEATURES].to_numpy(dtype=float),
            df.iloc[start+1:stop][TEMPERATURES].to_numpy(dtype=float))


def split_data(df, days):
    if days not in (3, 5, 10):
        raise ValueError("Training budget must be 3, 5, or 10 days")
    return (transitions(df, TRAIN_END-24*days, TRAIN_END),
            transitions(df, TRAIN_END, VALID_END),
            transitions(df, VALID_END, len(df)))


def describe_data(df, path):
    return {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "rows": len(df), "features": FEATURES, "targets": TEMPERATURES,
            "start": str(df.Time.iloc[0]), "end": str(df.Time.iloc[-1]),
            "split_observations": {"train": [0, 240], "validation": [240, 288],
                                   "test": [288, 385]},
            "train_pairs": {str(d): 24*d-1 for d in (3, 5, 10)},
            "validation_pairs": 47, "test_pairs": 96,
            "test_24h_origins": 73, "split_bounds_are": "zero-based, stop-exclusive"}
