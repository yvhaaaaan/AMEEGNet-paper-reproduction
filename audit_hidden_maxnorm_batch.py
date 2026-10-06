import argparse
import json
import math
from pathlib import Path

import numpy as np


def close(a, b, tol=1e-12):
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--subjects", nargs="+", default=[f"A{i:02d}" for i in range(1, 10)])
    parser.add_argument("--allow-imbalanced-test", action="store_true")
    args = parser.parse_args()
    root = Path(args.results)
    rows = []
    errors = []
    configs = []
    source_hashes = []

    for subject in args.subjects:
        json_path = root / f"{subject}.json"
        npz_path = root / f"{subject}.npz"
        pt_path = root / f"{subject}.pt"
        for path in (json_path, npz_path, pt_path):
            if not path.exists():
                errors.append(f"missing: {path.name}")
        if not json_path.exists() or not npz_path.exists():
            continue
        result = json.loads(json_path.read_text(encoding="utf-8"))
        configs.append({k: result.get(k) for k in (
            "version", "git_commit", "git_dirty", "input_scale",
            "input_normalization", "strict", "paper_pooling", "dropout",
            "dropout_after_pool", "head_dropout", "max_norm", "head_elu",
            "hidden_max_norm", "fusion", "eca", "bn_first", "eca_bias",
            "spatial_max_norm", "classifier_max_norm", "hidden_max_norm_value",
            "init_mode", "eca_stage", "fixed_fusion_channels", "reverse_sessions",
            "select_best_validation", "training_protocol",
            "norm_then_activation", "bn_eps", "bn_momentum", "seed", "epochs",
            "device", "deterministic", "test_evaluation", "checkpoint_selection",
            "softmax_before_loss", "fusion_pre_activation", "loader_rng",
            "conv_bias",
        )})
        source_hashes.append(result.get("source_sha256"))
        history = result.get("history", [])
        if len(history) != 1000:
            errors.append(f"{subject}: history length {len(history)}")
        if any(not math.isfinite(float(h["train_loss"])) for h in history):
            errors.append(f"{subject}: non-finite training loss")
        if any(h.get("test_acc") is not None for h in history):
            errors.append(f"{subject}: target test evaluated during training")
        if result.get("test_evaluations") != 1:
            errors.append(f"{subject}: test_evaluations={result.get('test_evaluations')}")
        if result.get("validation_fraction") is not None:
            errors.append(f"{subject}: unexpected validation split")
        with np.load(npz_path, allow_pickle=False) as pred:
            required = {"y_true", "y_pred", "fit_indices", "validation_indices",
                        "validation_y_true", "validation_y_pred"}
            if set(pred.files) != required:
                errors.append(f"{subject}: unexpected prediction keys {pred.files}")
            y_true = np.asarray(pred["y_true"])
            y_pred = np.asarray(pred["y_pred"])
            test_samples = int(result["training_protocol"]["test_samples"])
            if y_true.shape != (test_samples,) or y_pred.shape != (test_samples,):
                errors.append(f"{subject}: prediction shape {y_true.shape}/{y_pred.shape}")
            if (not args.allow_imbalanced_test and
                    not np.array_equal(np.unique(y_true, return_counts=True)[1], np.array([72] * 4))):
                errors.append(f"{subject}: target labels are not balanced")
            npz_acc = float(np.mean(y_true == y_pred))
            if not close(npz_acc, result["final_test_acc"]):
                errors.append(f"{subject}: JSON/NPZ accuracy mismatch")
        rows.append({
            "subject": subject,
            "accuracy": float(result["final_test_acc"]),
            "seconds": float(result["seconds"]),
            "train_samples": result["training_protocol"]["fit_samples"],
            "test_samples": result["training_protocol"]["test_samples"],
        })

    if len(configs) != len(rows):
        errors.append("not all subject configurations were loaded")
    if configs and any(config != configs[0] for config in configs[1:]):
        errors.append("subject configurations differ")
    if source_hashes and any(hashes != source_hashes[0] for hashes in source_hashes[1:]):
        errors.append("source hashes differ")
    accuracies = np.array([row["accuracy"] for row in rows], dtype=float)
    runtimes = np.array([row["seconds"] for row in rows], dtype=float)
    mean = float(accuracies.mean()) if len(accuracies) else float("nan")
    std = float(accuracies.std(ddof=1)) if len(accuracies) > 1 else float("nan")
    summary_path = root / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not close(summary.get("mean_final_acc"), mean):
            errors.append("summary mean mismatch")
        if not close(summary.get("std_final_acc"), std):
            errors.append("summary sample SD mismatch")
    report = {
        "results": str(root.resolve()),
        "subjects": rows,
        "n": len(rows),
        "mean_accuracy": mean,
        "sample_std_accuracy": std,
        "paper_mean": 0.8117,
        "difference_from_paper_pp": (mean - 0.8117) * 100,
        "runtime_mean_seconds": float(runtimes.mean()) if len(runtimes) else float("nan"),
        "runtime_median_seconds": float(np.median(runtimes)) if len(runtimes) else float("nan"),
        "runtime_total_seconds": float(runtimes.sum()) if len(runtimes) else float("nan"),
        "config": configs[0] if configs else None,
        "source_sha256": source_hashes[0] if source_hashes else None,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
