import argparse
import json
import math
from pathlib import Path

import numpy as np


EPOCHS = {
    "A01": 309, "A02": 121, "A03": 963, "A04": 346, "A05": 293,
    "A06": 571, "A07": 321, "A08": 147, "A09": 273,
}


def close(a, b, tol=1e-12):
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def comparable(result):
    keys = (
        "version", "git_commit", "git_dirty", "input_scale",
        "input_normalization", "strict", "paper_pooling", "dropout",
        "dropout_after_pool", "head_dropout", "max_norm", "head_elu",
        "hidden_max_norm", "fusion", "eca", "bn_first", "eca_bias",
        "spatial_max_norm", "classifier_max_norm", "hidden_max_norm_value",
        "init_mode", "eca_stage", "fixed_fusion_channels", "reverse_sessions",
        "select_best_validation", "norm_then_activation", "bn_eps",
        "bn_momentum", "seed", "device", "deterministic", "test_evaluation",
        "checkpoint_selection", "softmax_before_loss", "fusion_pre_activation",
        "loader_rng", "conv_bias",
    )
    values = {key: result.get(key) for key in keys}
    protocol = dict(result["training_protocol"])
    protocol.pop("fit_samples", None)
    protocol.pop("validation_samples", None)
    protocol.pop("test_samples", None)
    values["training_protocol"] = protocol
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    root = Path(args.results)
    errors = []
    rows = []
    configs = []
    hashes = []

    for subject, expected_epochs in EPOCHS.items():
        json_path = root / f"{subject}.json"
        npz_path = root / f"{subject}.npz"
        pt_path = root / f"{subject}.pt"
        for path in (json_path, npz_path, pt_path):
            if not path.exists():
                errors.append(f"{subject}: missing {path.name}")
        if not json_path.exists() or not npz_path.exists():
            continue
        result = json.loads(json_path.read_text(encoding="utf-8"))
        configs.append(comparable(result))
        hashes.append(result.get("source_sha256"))
        history = result.get("history", [])
        if result.get("epochs") != expected_epochs:
            errors.append(f"{subject}: epochs={result.get('epochs')}")
        if len(history) != expected_epochs:
            errors.append(f"{subject}: history length {len(history)}")
        if any(item.get("test_acc") is not None for item in history):
            errors.append(f"{subject}: target evaluated during training")
        if result.get("test_evaluations") != 1:
            errors.append(f"{subject}: test_evaluations={result.get('test_evaluations')}")
        protocol = result["training_protocol"]
        if protocol.get("fit_samples") != 288 or protocol.get("validation_samples") is not None:
            errors.append(f"{subject}: not a full-source fit")
        with np.load(npz_path, allow_pickle=False) as pred:
            y_true = np.asarray(pred["y_true"])
            y_pred = np.asarray(pred["y_pred"])
            if y_true.shape != (288,) or y_pred.shape != (288,):
                errors.append(f"{subject}: prediction shape {y_true.shape}/{y_pred.shape}")
            if not np.array_equal(np.bincount(y_true, minlength=4), np.array([72] * 4)):
                errors.append(f"{subject}: target labels are not balanced")
            if not close(np.mean(y_true == y_pred), result["final_test_acc"]):
                errors.append(f"{subject}: JSON/NPZ accuracy mismatch")
        rows.append({
            "subject": subject,
            "epochs": expected_epochs,
            "accuracy": float(result["final_test_acc"]),
            "seconds": float(result["seconds"]),
        })

    if len(configs) != len(EPOCHS):
        errors.append("not all subjects were loaded")
    if configs and any(config != configs[0] for config in configs[1:]):
        errors.append("common configurations differ")
    if hashes and any(item != hashes[0] for item in hashes[1:]):
        errors.append("source hashes differ")

    accuracies = np.array([row["accuracy"] for row in rows], dtype=float)
    runtimes = np.array([row["seconds"] for row in rows], dtype=float)
    mean = float(accuracies.mean())
    std = float(accuracies.std(ddof=1))
    report = {
        "results": str(root.resolve()), "subjects": rows, "n": len(rows),
        "mean_accuracy": mean, "sample_std_accuracy": std,
        "paper_mean": 0.8117,
        "difference_from_paper_pp": (mean - 0.8117) * 100,
        "previous_best_mean": 0.786651234568,
        "difference_from_previous_best_pp": (mean - 0.786651234568) * 100,
        "runtime_mean_seconds": float(runtimes.mean()),
        "runtime_median_seconds": float(np.median(runtimes)),
        "runtime_total_seconds": float(runtimes.sum()),
        "config": configs[0] if configs else None,
        "source_sha256": hashes[0] if hashes else None,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
