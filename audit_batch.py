import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reference-mean", type=float, default=81.17)
    parser.add_argument("--reference-sd", type=float, default=10.43)
    args = parser.parse_args()

    rows, hashes, configs = [], [], []
    for i in range(1, 10):
        sid = f"A{i:02d}"
        path = args.directory / f"{sid}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["epochs"] == 1000
        assert record["test_evaluation"] == "final"
        assert record["test_evaluations"] == 1
        history = record["history"]
        assert [h["epoch"] for h in history] == list(range(1, 1001))
        assert all(np.isfinite(h["train_loss"]) for h in history)
        assert all(h["test_acc"] is None for h in history)
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as pred:
            truth, predicted = pred["y_true"], pred["y_pred"]
            assert truth.shape == predicted.shape == (288,)
            assert np.array_equal(np.bincount(truth, minlength=4), [72] * 4)
            assert np.isin(predicted, range(4)).all()
            accuracy = float(np.mean(truth == predicted))
        assert np.isclose(accuracy, record["final_test_acc"])
        configs.append({k: record[k] for k in (
            "version", "git_commit", "git_dirty", "strict", "paper_pooling",
            "dropout", "dropout_after_pool", "max_norm", "head_elu", "fusion",
            "eca", "bn_first", "norm_then_activation", "seed", "epochs",
            "device", "deterministic", "test_evaluation", "training_protocol")})
        rows.append((sid, accuracy * 100, record["seconds"]))
        for suffix in (".json", ".npz", ".pt"):
            artifact = path.with_suffix(suffix)
            assert artifact.stat().st_size > 0
            hashes.append((artifact.name, sha256(artifact)))

    assert all(c == configs[0] for c in configs)
    values = np.array([row[1] for row in rows])
    seconds = np.array([row[2] for row in rows])
    summary = json.loads((args.directory / "summary.json").read_text(encoding="utf-8"))
    assert np.isclose(values.mean() / 100, summary["mean_final_acc"])
    assert np.isclose(values.std(ddof=1) / 100, summary["std_final_acc"])
    lines = ["# Nine-subject standard EEGNet branch audit", "",
             "All nine records completed 1000 epochs with the same v1.0.17 "
             "configuration. The held-out E session was evaluated once after "
             "training and was not used for selection.", "",
             "| Subject | Final E accuracy (%) | Runtime (s) |", 
             "| --- | ---: | ---: |"]
    lines += [f"| {sid} | {acc:.2f} | {sec:.2f} |" for sid, acc, sec in rows]
    lines += ["", f"Mean +/- sample SD: {values.mean():.2f}% +/- {values.std(ddof=1):.2f}%.",
              f"Difference from paper mean {args.reference_mean:.2f}%: "
              f"{values.mean() - args.reference_mean:+.2f} percentage points.",
              f"Difference from paper SD {args.reference_sd:.2f}%: "
              f"{values.std(ddof=1) - args.reference_sd:+.2f} percentage points.",
              f"Runtime mean: {seconds.mean():.2f} s; median: {np.median(seconds):.2f} s.",
              f"Runtime range: {seconds.min():.2f}--{seconds.max():.2f} s.", "",
              "This is a controlled standard EEGNet pooling/dropout hypothesis, "
              "not a confirmed complete original-paper reproduction. The paper "
              "does not fully specify pooling, dropout, padding, initialization, "
              "or epoch selection. No v1.7 enhancement was used.", "",
              "## Configuration", "", "```json", json.dumps(configs[0], indent=2),
              "```", "", "## Local artifact SHA-256", "",
              "| File | SHA-256 |", "| --- | --- |"]
    lines += [f"| {name} | {digest} |" for name, digest in hashes]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PASS: 9 subjects; {values.mean():.2f}% +/- {values.std(ddof=1):.2f}%")


if __name__ == "__main__":
    main()
