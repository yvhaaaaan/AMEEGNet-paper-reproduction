from pathlib import Path
import numpy as np


def load_subject_npz(path):
    path = str(path)
    with np.load(path, allow_pickle=False) as z:
        x = np.asarray(z["x"], dtype=np.float32)
        y = np.asarray(z["y"], dtype=np.int64)
        sessions = np.asarray(z["sessions"]).astype(str)
    if x.shape != (576, 22, 1125) or y.shape != (576,) or sessions.shape != (576,):
        raise ValueError(f"expected (576,22,1125), got {x.shape}, {y.shape}, {sessions.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"non-finite EEG values in {path}")
    if not np.isin(y, np.arange(4)).all():
        raise ValueError(f"labels outside 0..3 in {path}")
    train = sessions == "0train"
    test = sessions == "1test"
    if train.sum() != 288 or test.sum() != 288:
        raise ValueError("expected 288 train and 288 test trials")
    if np.any(train & test) or np.any(~(train | test)):
        raise ValueError("sessions must be exactly 0train or 1test")
    return x[train], y[train], x[test], y[test]


def session_standardize(x_train, x_test, eps=1e-6):
    """Fit channel statistics on training trials, then transform both sessions."""
    mu = x_train.mean(axis=(0, 2), keepdims=True)
    sd = x_train.std(axis=(0, 2), keepdims=True).clip(min=eps)
    return ((x_train - mu) / sd).astype(np.float32), ((x_test - mu) / sd).astype(np.float32)
