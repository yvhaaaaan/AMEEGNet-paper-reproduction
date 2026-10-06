import argparse
from pathlib import Path

import numpy as np
import scipy.io


def load_session(path):
    mat = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False)
    data = mat["data"]
    xs, ys = [], []
    for idx in range(data.shape[1]):
        run = data[0, idx].item()
        X = np.asarray(run.X, dtype=np.float32)
        trials = np.asarray(run.trial, dtype=np.int64).ravel()
        labels = np.asarray(run.y, dtype=np.int64).ravel()
        artifacts = np.asarray(run.artifacts, dtype=np.int64).ravel()
        if not len(trials):
            continue
        if X.ndim != 2 or X.shape[1] < 22:
            raise ValueError(f"unexpected signal shape in {path}: {X.shape}")
        if not (len(trials) == len(labels) == len(artifacts)):
            raise ValueError(f"run fields have different lengths in {path}")
        if trials.min() >= 1:
            trials = trials - 1
        X = X[:, :22].T
        keep = (artifacts == 0) & np.isin(labels, [1, 2, 3, 4])
        for onset, label in zip(trials[keep], labels[keep]):
            start = int(onset) + 375
            stop = start + 1125
            if start < 0 or stop > X.shape[1]:
                raise ValueError(f"window out of bounds in {path}: {start}:{stop}")
            xs.append(X[:, start:stop])
            ys.append(int(label) - 1)
    if not xs:
        raise ValueError(f"no clean labeled trials in {path}")
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    train, y_train = load_session(args.raw.with_name(args.raw.stem + "T.mat"))
    test, y_test = load_session(args.raw.with_name(args.raw.stem + "E.mat"))
    x = np.concatenate([train, test])
    y = np.concatenate([y_train, y_test])
    sessions = np.array(["0train"] * len(y_train) + ["1test"] * len(y_test))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, x=x, y=y, sessions=sessions)
    print({"out": str(args.out), "train": len(y_train), "test": len(y_test),
           "train_counts": np.bincount(y_train, minlength=4).tolist(),
           "test_counts": np.bincount(y_test, minlength=4).tolist()})


if __name__ == "__main__":
    main()
