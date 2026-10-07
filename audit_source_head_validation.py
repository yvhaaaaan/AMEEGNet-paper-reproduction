"""Audit the registered paired source-validation screen, not target results."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from audit_hidden_maxnorm_batch import comparable_config


SUBJECTS = ("A01", "A02", "A04", "A06")
SEEDS = (1, 7, 2025)


def check_source_result(result, predictions, expected_head_elu, seed):
    errors = []
    if result.get("seed") != seed or result.get("head_elu") != expected_head_elu:
        errors.append("seed or hidden activation differs from registration")
    if result.get("git_dirty") or result.get("checkpoint_selection") != "final_epoch":
        errors.append("dirty training checkout or non-final checkpoint")
    if result.get("test_evaluation") != "none" or result.get("test_evaluations") != 0:
        errors.append("target evaluation is not disabled")
    if result.get("final_test_acc") is not None:
        errors.append("unexpected target score")
    protocol = result.get("training_protocol", {})
    if (protocol.get("fit_samples"), protocol.get("validation_samples")) != (230, 58):
        errors.append("incorrect 230/58 source split")
    if result.get("validation_fraction") != 0.2:
        errors.append("incorrect validation fraction")
    history = result.get("history", [])
    if result.get("epochs") != 1000 or [h.get("epoch") for h in history] != list(range(1, 1001)):
        errors.append("incomplete 1000-epoch history")
    for item in history:
        if item.get("test_acc") is not None:
            errors.append("target evaluated in history")
            break
        for key in ("train_loss", "train_acc", "val_loss", "val_acc"):
            value = item.get(key)
            if value is None or not math.isfinite(float(value)):
                errors.append(f"missing/non-finite history {key}")
                break
            if key.endswith("acc") and not 0 <= float(value) <= 1:
                errors.append(f"history {key} outside [0,1]")
            if key.endswith("loss") and float(value) < 0:
                errors.append(f"negative history {key}")
    if predictions["y_true"].size or predictions["y_pred"].size:
        errors.append("target predictions present")
    fit = predictions["fit_indices"]
    val = predictions["validation_indices"]
    if fit.shape != (230,) or val.shape != (58,):
        errors.append("wrong source index dimensions")
    elif not np.array_equal(np.sort(np.concatenate((fit, val))), np.arange(288)):
        errors.append("source indices do not partition 288 trials")
    true = predictions["validation_y_true"]
    predicted = predictions["validation_y_pred"]
    if true.shape != (58,) or predicted.shape != (58,):
        errors.append("wrong validation prediction dimensions")
    elif not np.isin(true, np.arange(4)).all() or not np.isin(predicted, np.arange(4)).all():
        errors.append("validation labels outside class range")
    elif not math.isclose(float(np.mean(true == predicted)), result.get("final_val_acc", -1),
                          rel_tol=1e-12, abs_tol=1e-12):
        errors.append("JSON/NPZ validation accuracy differs")
    if history and not math.isclose(float(history[-1].get("val_acc", -1)),
                                    float(result.get("final_val_acc", -1)), abs_tol=1e-12):
        errors.append("final validation differs from the final epoch")
    return errors


def load_run(path, head_elu, seed):
    for extension in ("json", "npz", "pt"):
        artifact = path.with_suffix(f".{extension}")
        if not artifact.exists() or not artifact.stat().st_size:
            raise ValueError(f"missing/empty artifact: {artifact}")
    result = json.loads(path.read_text(encoding="utf-8"))
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
        predictions = {key: saved[key].copy() for key in saved.files}
    errors = check_source_result(result, predictions, head_elu, seed)
    return result, predictions, errors


def summarize(root, references):
    rows = []
    errors = []
    all_settings = []
    all_hashes = []
    for subject in SUBJECTS:
        for seed in SEEDS:
            filename = f"{subject}_seed{seed}.json"
            control_path = (root / "elu" if subject == "A06" else references) / filename
            changed_path = root / "noelu" / filename
            control, c_pred, c_errors = load_run(control_path, True, seed)
            changed, n_pred, n_errors = load_run(changed_path, False, seed)
            errors.extend(f"{subject}/seed{seed}/ELU: {e}" for e in c_errors)
            errors.extend(f"{subject}/seed{seed}/noELU: {e}" for e in n_errors)
            for key in ("fit_indices", "validation_indices", "validation_y_true"):
                if not np.array_equal(c_pred[key], n_pred[key]):
                    errors.append(f"{subject}/seed{seed}: paired {key} differs")
            if control.get("data_sha256") != changed.get("data_sha256"):
                errors.append(f"{subject}/seed{seed}: paired input hashes differ")
            for run in (control, changed):
                settings = comparable_config(run)
                for key in ("version", "git_commit", "head_elu", "seed"):
                    settings.pop(key)
                all_settings.append(settings)
                all_hashes.append(run.get("source_sha256"))
            correct_delta = int(np.sum(n_pred["validation_y_true"] == n_pred["validation_y_pred"])) - int(
                np.sum(c_pred["validation_y_true"] == c_pred["validation_y_pred"]))
            rows.append({
                "subject": subject, "seed": seed,
                "elu_final_val": control["final_val_acc"],
                "noelu_final_val": changed["final_val_acc"],
                "delta_correct_trials": correct_delta,
                "delta_pp": correct_delta * 100 / 58,
                "noelu_final_train_acc": changed["history"][-1]["train_acc"],
                "control_path": str(control_path.resolve()),
                "changed_path": str(changed_path.resolve()),
            })
    if any(settings != all_settings[0] for settings in all_settings[1:]):
        errors.append("algorithm settings differ beyond hidden ELU and seed")
    if not all_hashes[0] or any(value != all_hashes[0] for value in all_hashes[1:]):
        errors.append("missing or differing training-source hashes")
    subject_rows = []
    for subject in SUBJECTS:
        selected = [row for row in rows if row["subject"] == subject]
        subject_rows.append({
            "subject": subject,
            "elu_seed_mean_val": float(np.mean([r["elu_final_val"] for r in selected])),
            "noelu_seed_mean_val": float(np.mean([r["noelu_final_val"] for r in selected])),
            "mean_delta_pp": sum(r["delta_correct_trials"] for r in selected) * 100 / (58 * len(SEEDS)),
        })
    total_correct_delta = sum(row["delta_correct_trials"] for row in rows)
    mean_delta = total_correct_delta * 100 / (58 * len(rows))
    stable = all(row["noelu_final_train_acc"] >= 0.95 for row in rows)
    return {
        "scope": "source_only_paired_validation_not_a_paper_result",
        "n_subjects": 4, "n_seeds": 3, "n_pairs": 12,
        "target_evaluations": 0, "pairs": rows, "subjects": subject_rows,
        "mean_elu_final_validation": float(np.mean([r["elu_seed_mean_val"] for r in subject_rows])),
        "mean_noelu_final_validation": float(np.mean([r["noelu_seed_mean_val"] for r in subject_rows])),
        "mean_paired_delta_pp": mean_delta,
        "subject_delta_sample_sd_pp": float(np.std([r["mean_delta_pp"] for r in subject_rows], ddof=1)),
        "all_noelu_final_train_acc_at_least_95_percent": stable,
        "paired_total_correct_trial_delta": total_correct_delta,
        "source_gate_pass": not errors and stable and total_correct_delta >= 0,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--references", default="results/seed_screen_v1_0_74", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = summarize(args.results, args.references)
    if report["errors"]:
        print(json.dumps(report, indent=2))
        raise SystemExit(1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
