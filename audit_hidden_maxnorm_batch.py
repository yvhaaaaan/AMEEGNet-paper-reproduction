import argparse
import json
import math
from pathlib import Path

import numpy as np


def close(a, b, tol=1e-12):
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def comparable_config(result):
    """Return run settings without subject-specific sample counts."""
    config = {
        k: result.get(k) for k in (
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
            "conv_bias", "validation_fraction",
        )
    }
    protocol = dict(config["training_protocol"] or {})
    for key in ("fit_samples", "validation_samples", "test_samples"):
        protocol.pop(key, None)
    config["training_protocol"] = protocol
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--subjects", nargs="+", default=[f"A{i:02d}" for i in range(1, 10)])
    parser.add_argument("--allow-imbalanced-test", action="store_true")
    parser.add_argument("--expected-fit-samples", type=int, default=None,
                        help="required source-fit count for the registered protocol")
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
        configs.append(comparable_config(result))
        source_hashes.append(result.get("source_sha256"))
        history = result.get("history", [])
        if len(history) != 1000:
            errors.append(f"{subject}: history length {len(history)}")
        if [h.get("epoch") for h in history] != list(range(1, 1001)):
            errors.append(f"{subject}: history epoch sequence is incomplete or duplicated")
        if any(not math.isfinite(float(h["train_loss"])) for h in history):
            errors.append(f"{subject}: non-finite training loss")
        if any(h.get("test_acc") is not None for h in history):
            errors.append(f"{subject}: target test evaluated during training")
        if result.get("test_evaluations") != 1:
            errors.append(f"{subject}: test_evaluations={result.get('test_evaluations')}")
        has_validation = result.get("validation_fraction") is not None
        with np.load(npz_path, allow_pickle=False) as pred:
            required = {"y_true", "y_pred", "fit_indices", "validation_indices",
                        "validation_y_true", "validation_y_pred"}
            if set(pred.files) != required:
                errors.append(f"{subject}: unexpected prediction keys {pred.files}")
            y_true = np.asarray(pred["y_true"])
            y_pred = np.asarray(pred["y_pred"])
            protocol = result["training_protocol"]
            train_samples = int(protocol["fit_samples"])
            test_samples = int(protocol["test_samples"])
            if args.expected_fit_samples is not None and train_samples != args.expected_fit_samples:
                errors.append(f"{subject}: fit_samples={train_samples}, expected {args.expected_fit_samples}")
            if y_true.shape != (test_samples,) or y_pred.shape != (test_samples,):
                errors.append(f"{subject}: prediction shape {y_true.shape}/{y_pred.shape}")
            if len(np.unique(y_true)) != 4:
                errors.append(f"{subject}: target labels do not contain all four classes")
            if not np.isin(y_pred, np.arange(4)).all():
                errors.append(f"{subject}: target predictions outside class range")
            if not args.allow_imbalanced_test:
                counts = np.bincount(y_true, minlength=4)
                if not np.array_equal(counts, np.array([72] * 4)):
                    errors.append(f"{subject}: target labels are not balanced")
            npz_acc = float(np.mean(y_true == y_pred))
            if not close(npz_acc, result["final_test_acc"]):
                errors.append(f"{subject}: JSON/NPZ accuracy mismatch")
            if pred["fit_indices"].shape != (train_samples,):
                errors.append(f"{subject}: fit index count mismatch")
            fit_indices = np.asarray(pred["fit_indices"], dtype=np.int64)
            validation_indices = np.asarray(pred["validation_indices"], dtype=np.int64)
            validation_y_true = np.asarray(pred["validation_y_true"], dtype=np.int64)
            validation_y_pred = np.asarray(pred["validation_y_pred"], dtype=np.int64)
            source_samples = train_samples
            if has_validation:
                validation_samples = int(protocol["validation_samples"])
                source_samples += validation_samples
                if validation_indices.shape != (validation_samples,):
                    errors.append(f"{subject}: validation index count mismatch")
                if validation_y_true.shape != (validation_samples,) or validation_y_pred.shape != (validation_samples,):
                    errors.append(f"{subject}: validation prediction shape mismatch")
                if validation_indices.size and not np.array_equal(
                    np.sort(np.concatenate([fit_indices, validation_indices])),
                    np.arange(source_samples, dtype=np.int64),
                ):
                    errors.append(f"{subject}: fit/validation indices do not partition source data")
                if validation_y_pred.size and not np.all((0 <= validation_y_pred) & (validation_y_pred < 4)):
                    errors.append(f"{subject}: validation predictions outside class range")
                if validation_y_true.size:
                    val_acc = float(np.mean(validation_y_true == validation_y_pred))
                    if not close(val_acc, result.get("final_val_acc")):
                        errors.append(f"{subject}: JSON/NPZ validation accuracy mismatch")
            else:
                if not np.array_equal(np.sort(fit_indices), np.arange(train_samples)):
                    errors.append(f"{subject}: fit indices do not cover full source data")
                if validation_indices.size != 0:
                    errors.append(f"{subject}: unexpected validation indices")
                if validation_y_true.size != 0 or validation_y_pred.size != 0:
                    errors.append(f"{subject}: unexpected validation predictions")
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
