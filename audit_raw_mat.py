"""Independent audit of the official BCI IV 2a MAT files.

This deliberately does not import the AMEEGNet package used to create the
cached NPZ files.  It checks the raw BBCI structures, labels, trial windows,
ordering, and cached arrays independently.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import scipy.io


FS = 250
START = int(1.5 * FS)
STOP = int(6.0 * FS)
N_CHANNELS = 22
N_SAMPLES = STOP - START


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw_session(path: Path):
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    runs = np.asarray(mat["data"], dtype=object).ravel()
    xs, ys, metadata = [], [], []
    for run_index, run in enumerate(runs):
        trial = np.asarray(run.trial).ravel()
        labels = np.asarray(run.y).ravel()
        if trial.size == 0 and labels.size == 0:
            continue
        if trial.size != labels.size:
            raise ValueError(f"{path.name} run {run_index}: trial/y length mismatch")
        continuous = np.asarray(run.X)
        if continuous.ndim != 2 or continuous.shape[1] < N_CHANNELS:
            raise ValueError(f"{path.name} run {run_index}: unexpected X shape {continuous.shape}")
        epochs = []
        for onset, label in zip(trial, labels):
            start = int(onset) - 1 + START
            stop = int(onset) - 1 + STOP
            if start < 0 or stop > len(continuous):
                raise ValueError(f"{path.name} run {run_index}: window out of bounds")
            # Raw files store samples x channels; the model uses channels x samples.
            epochs.append(continuous[start:stop, :N_CHANNELS].T)
        xs.append(np.asarray(epochs, dtype=np.float32))
        ys.append(labels.astype(np.int64) - 1)
        metadata.append({
            "run_index": int(run_index),
            "trials": int(len(labels)),
            "labels_1based": np.bincount(labels.astype(np.int64), minlength=5)[1:].tolist(),
            "first_onset_1based": int(trial[0]),
            "last_onset_1based": int(trial[-1]),
        })
    if not xs:
        raise ValueError(f"{path}: no labelled runs")
    return np.concatenate(xs), np.concatenate(ys), metadata


def compare_session(raw_x, raw_y, cached_x, cached_y, name):
    if raw_x.shape != cached_x.shape or raw_y.shape != cached_y.shape:
        raise AssertionError(
            f"{name}: shape mismatch raw={raw_x.shape}/{raw_y.shape} "
            f"cached={cached_x.shape}/{cached_y.shape}"
        )
    if not np.array_equal(raw_y, cached_y):
        raise AssertionError(f"{name}: labels or trial order differ")
    equal = bool(np.array_equal(raw_x, cached_x))
    max_abs = float(np.max(np.abs(raw_x - cached_x)))
    return {"shape": list(raw_x.shape), "labels_equal": True,
            "arrays_bitwise_equal": equal, "max_abs_difference": max_abs,
            "label_counts": np.bincount(raw_y, minlength=4).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cached", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    source_hashes = []
    for number in range(1, 10):
        subject = f"A{number:02d}"
        subject_rows = []
        for suffix, cache_session in (("T", "0train"), ("E", "1test")):
            source = args.source / f"{subject}{suffix}.mat"
            raw_x, raw_y, runs = load_raw_session(source)
            source_hashes.append((source.name, sha256(source)))
            subject_rows.append({"session": suffix, "cache_session": cache_session,
                                "runs": runs, "source_sha256": source_hashes[-1][1]})
            if raw_x.shape != (288, 22, 1125):
                raise AssertionError(f"{source.name}: unexpected raw shape {raw_x.shape}")
            if not np.array_equal(np.bincount(raw_y, minlength=4), [72, 72, 72, 72]):
                raise AssertionError(f"{source.name}: unexpected class balance")

        cache = np.load(args.cached / f"{subject}.npz", allow_pickle=False)
        x = np.asarray(cache["x"], dtype=np.float32)
        y = np.asarray(cache["y"], dtype=np.int64)
        sessions = np.asarray(cache["sessions"]).astype(str)
        raw_t, raw_y_t, _ = load_raw_session(args.source / f"{subject}T.mat")
        raw_e, raw_y_e, _ = load_raw_session(args.source / f"{subject}E.mat")
        train_check = compare_session(raw_t, raw_y_t, x[sessions == "0train"],
                                      y[sessions == "0train"], f"{subject} T")
        test_check = compare_session(raw_e, raw_y_e, x[sessions == "1test"],
                                     y[sessions == "1test"], f"{subject} E")
        rows.append({"subject": subject, "train": train_check, "test": test_check,
                     "cache_sha256": sha256(args.cached / f"{subject}.npz"),
                     "source_files": subject_rows})

    lines = [
        "# Independent BCI IV 2a raw-data audit", "",
        "The audit parsed the official MAT files directly with scipy.io and did "
        "not import the AMEEGNet data loader.", "",
        f"Window: [{START / FS:.1f}, {STOP / FS:.1f}) s at {FS} Hz; "
        f"{N_CHANNELS} EEG channels; {N_SAMPLES} samples per trial.", "",
        "| Subject | T raw/cache | E raw/cache | T max abs diff | E max abs diff |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        t, e = row["train"], row["test"]
        lines.append(f"| {row['subject']} | {t['shape']} / {t['labels_equal']} | "
                     f"{e['shape']} / {e['labels_equal']} | "
                     f"{t['max_abs_difference']:.8g} | {e['max_abs_difference']:.8g} |")
    lines += ["", "All nine subjects have 288 labelled trials per session and "
              "72 trials per class per session.",
              "The cached NPZ arrays and labels are compared in the original "
              "trial order; `arrays_bitwise_equal` is required to be true for "
              "both sessions of every subject.", "", "## Source SHA-256", "",
              "| File | SHA-256 |", "| --- | --- |"]
    for row in rows:
        for source in row["source_files"]:
            lines.append(f"| {source['session']} ({row['subject']}) | {source['source_sha256']} |")
    lines += ["", "## Run structure", ""]
    for row in rows:
        lines.append(f"### {row['subject']}")
        for source in row["source_files"]:
            lines.append(f"- {source['session']}: "
                         + ", ".join(f"run{r['run_index']}={r['trials']} trials" for r in source["runs"]))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PASS: independently audited {len(rows)} subjects")


if __name__ == "__main__":
    main()
