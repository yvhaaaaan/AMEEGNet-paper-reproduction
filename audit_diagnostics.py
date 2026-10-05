import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ameegnet_ours import AMEEGNet, load_subject_npz


RUNS = (
    ("default_bn", "bn_diag_eps1e-5_m01", 1e-5, 0.1, 0.5, "1.0.26"),
    ("epsilon_only", "bn_diag_eps1e-3_m01", 1e-3, 0.1, 0.5, "1.0.26"),
    ("momentum_only", "bn_diag_eps1e-5_m001", 1e-5, 0.01, 0.5, "1.0.26"),
    ("both_bn_changes", "bn_diag_eps1e-3_m001", 1e-3, 0.01, 0.5, "1.0.26"),
    ("head_dropout_off", "head_dropout_off_a01", 1e-5, 0.1, 0.0, "1.0.27"),
)
BUFFER_SUFFIXES = (".running_mean", ".running_var", ".num_batches_tracked")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(state):
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def evaluate_validation(state, record, x_train, y_train, indices, device):
    model = AMEEGNet(
        pool=record["paper_pooling"], dropout=record["dropout"],
        bn_first=record["bn_first"],
        norm_then_activation=record["norm_then_activation"],
        fusion=record["fusion"], eca=record["eca"],
        dropout_after_pool=record["dropout_after_pool"],
        head_elu=record["head_elu"],
        head_dropout=record.get("head_dropout", record["dropout"]),
        bn_eps=record["bn_eps"], bn_momentum=record["bn_momentum"],
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    bn_layers = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    require(len(bn_layers) == 6, "Expected six non-temporal BatchNorm layers")
    for module in bn_layers:
        require(int(module.num_batches_tracked) == 4000, "Unexpected BN update count")
        require(torch.isfinite(module.running_mean).all().item(), "Non-finite BN mean")
        require((torch.isfinite(module.running_var) & (module.running_var >= 0)).all().item(),
                "Invalid BN variance")
    before = fingerprint(model.state_dict())
    inputs = torch.from_numpy(np.asarray(x_train[indices], dtype=np.float32)).to(device)
    truth = torch.from_numpy(np.asarray(y_train[indices], dtype=np.int64)).to(device)
    with torch.inference_mode():
        logits = model(inputs)
        repeated = model(inputs)
        require(torch.equal(logits, repeated), "Repeated validation evaluation changed")
        require(torch.isfinite(logits).all().item(), "Non-finite validation logits")
        loss = float(nn.functional.cross_entropy(logits, truth).item())
        predicted = logits.argmax(1).cpu().numpy()
    require(before == fingerprint(model.state_dict()), "Evaluation changed model state")
    return loss, predicted


def audit(directory, device):
    directory = directory.resolve()
    repository = Path(__file__).resolve().parent
    rows, states, histories, artifacts = [], {}, {}, []
    reference = None
    for label, stem, eps, momentum, head_dropout, version in RUNS:
        path = directory / (stem + ".json")
        record = json.loads(path.read_text(encoding="utf-8"))
        require(record["version"] == version and record["git_dirty"] is False,
                f"{stem}: unexpected version or dirty source")
        tag_commit = subprocess.check_output(
            ["git", "rev-parse", f"v{version}^{{commit}}"], cwd=repository, text=True
        ).strip()
        require(record["git_commit"] == tag_commit, f"{stem}: tag/commit mismatch")
        require(record["bn_eps"] == eps and record["bn_momentum"] == momentum,
                f"{stem}: wrong BN factors")
        require(record.get("head_dropout", record["dropout"]) == head_dropout,
                f"{stem}: wrong head dropout")
        config = {key: record[key] for key in (
            "strict", "paper_pooling", "dropout", "dropout_after_pool", "max_norm",
            "head_elu", "reverse_sessions", "fusion", "eca", "bn_first",
            "norm_then_activation", "validation_fraction", "seed", "epochs",
            "deterministic", "initial_state_sha256", "training_protocol", "runtime",
        )}
        require(config["strict"] and config["paper_pooling"] and config["max_norm"],
                f"{stem}: architecture/protocol changed")
        require(config["dropout"] == 0.5 and config["dropout_after_pool"],
                f"{stem}: branch dropout changed")
        require(config["fusion"] and config["eca"] and config["head_elu"] and
                not config["bn_first"] and not config["reverse_sessions"] and
                config["norm_then_activation"], f"{stem}: unexpected architecture")
        require(config["seed"] == 42 and config["epochs"] == 1000 and
                config["validation_fraction"] == 0.2 and config["deterministic"],
                f"{stem}: seed/split/epochs changed")
        require(record["test_evaluation"] == "none" and record["test_evaluations"] == 0
                and record["final_test_acc"] is None, f"{stem}: target Session evaluated")
        require(record["checkpoint_selection"] == "final_epoch", f"{stem}: epoch selection changed")
        require(record["training_protocol"] == {
            "optimizer": "Adam", "lr": 0.001, "weight_decay": 0.0,
            "betas": [0.9, 0.999], "eps": 1e-8, "gradient_clip_max_norm": None,
            "batch_size": 64, "fit_samples": 230, "validation_samples": 58,
            "test_samples": 288, "label_smoothing": 0.0,
        }, f"{stem}: optimizer or sample counts changed")
        data_path = Path(record["data_path"])
        require(sha256(data_path) == record["data_sha256"], f"{stem}: data hash mismatch")
        history = record["history"]
        require([item["epoch"] for item in history] == list(range(1, 1001)),
                f"{stem}: incomplete history")
        for item in history:
            require(item["test_acc"] is None, f"{stem}: in-training target evaluation")
            for key in ("train_loss", "val_loss", "gradient_norm_mean", "gradient_norm_max"):
                require(math.isfinite(item[key]) and item[key] >= 0,
                        f"{stem}: invalid {key}")
            for key in ("train_acc", "val_acc"):
                require(math.isfinite(item[key]) and 0 <= item[key] <= 1,
                        f"{stem}: invalid {key}")
        best = max(history, key=lambda item: item["val_acc"])
        require(best["epoch"] == record["best_epoch"] and
                best["val_acc"] == record["best_val_acc"], f"{stem}: best validation mismatch")
        require(history[-1]["val_acc"] == record["final_val_acc"] and
                history[-1]["val_loss"] == record["final_val_loss"],
                f"{stem}: final validation mismatch")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as predictions:
            fit, val = predictions["fit_indices"], predictions["validation_indices"]
            require(fit.shape == (230,) and val.shape == (58,), f"{stem}: wrong split size")
            require(np.array_equal(np.sort(np.concatenate((fit, val))), np.arange(288)),
                    f"{stem}: split overlap or omissions")
            require(predictions["y_true"].size == predictions["y_pred"].size == 0,
                    f"{stem}: saved target predictions")
            truth, saved_pred = predictions["validation_y_true"], predictions["validation_y_pred"]
            require(truth.shape == saved_pred.shape == (58,) and
                    np.isin(saved_pred, range(4)).all(), f"{stem}: invalid predictions")
            require(np.isclose(np.mean(truth == saved_pred), record["final_val_acc"]),
                    f"{stem}: prediction accuracy mismatch")
            # Load only the source Session arrays during this independent audit.
            x_train, y_train, _, _ = load_subject_npz(data_path)
            require(np.array_equal(truth, y_train[val]), f"{stem}: wrong validation labels")
            checkpoint = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=False)
            checkpoint_result = json.loads(json.dumps(checkpoint["result"]))
            require(checkpoint_result == record, f"{stem}: JSON/checkpoint mismatch")
            state = checkpoint["model"]
            require(fingerprint(state) == record["final_state_sha256"],
                    f"{stem}: final state fingerprint mismatch")
            loss, predicted = evaluate_validation(state, record, x_train, y_train, val, device)
            require(np.array_equal(predicted, saved_pred) and
                    np.isclose(loss, record["final_val_loss"], rtol=1e-5, atol=1e-6),
                    f"{stem}: saved model validation mismatch")
            shared = {"config": config, "data_hash": record["data_sha256"],
                      "fit": fit.tolist(), "val": val.tolist(), "truth": truth.tolist()}
            if reference is None:
                reference = shared
            require(shared == reference, f"{stem}: comparison is not controlled")
            confusion = np.zeros((4, 4), dtype=int)
            np.add.at(confusion, (truth, saved_pred), 1)
        require(all(torch.isfinite(tensor).all().item() for tensor in state.values()),
                f"{stem}: non-finite model state")
        states[label] = state
        histories[label] = history
        inventory = [path, path.with_suffix(".pt"), path.with_suffix(".npz")]
        checkpoint_dir = directory / (stem + "_checkpoints")
        require(sorted(item.name for item in checkpoint_dir.glob("*.pt")) ==
                [f"epoch_{epoch:04d}.pt" for epoch in (250, 500, 750, 1000)],
                f"{stem}: checkpoint inventory mismatch")
        for epoch in (250, 500, 750, 1000):
            checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            require(saved["epoch"] == epoch and saved["git_commit"] == record["git_commit"]
                    and saved["data_sha256"] == record["data_sha256"],
                    f"{stem}: checkpoint provenance mismatch")
            require(np.array_equal(saved["fit_indices"], fit) and
                    np.array_equal(saved["validation_indices"], val),
                    f"{stem}: checkpoint split changed")
            require(all(torch.isfinite(value).all().item() for value in saved["model"].values()),
                    f"{stem}: non-finite periodic checkpoint")
            if epoch == 1000:
                require(fingerprint(saved["model"]) == record["final_state_sha256"],
                        f"{stem}: periodic/final checkpoint mismatch")
            inventory.append(checkpoint_path)
        artifacts.extend({"path": str(item), "bytes": item.stat().st_size,
                          "sha256": sha256(item)} for item in inventory)
        rows.append({
            "label": label, "file": path.name, "version": version,
            "git_commit": record["git_commit"], "source_sha256": record["source_sha256"],
            "bn_eps": eps, "bn_momentum": momentum, "head_dropout": head_dropout,
            "final_validation_accuracy": record["final_val_acc"],
            "final_validation_loss": record["final_val_loss"],
            "best_validation_accuracy": record["best_val_acc"], "best_epoch": record["best_epoch"],
            "late_validation_mean": float(np.mean([item["val_acc"] for item in history[-100:]])),
            "max_gradient_norm": max(item["gradient_norm_max"] for item in history),
            "seconds": record["seconds"], "validation_confusion_matrix": confusion.tolist(),
            "trained_parameters_sha256": fingerprint({
                name: tensor for name, tensor in state.items() if not name.endswith(BUFFER_SUFFIXES)
            }),
        })

    bn_rows = [row for row in rows if row["version"] == "1.0.26"]
    require(all(row["source_sha256"] == bn_rows[0]["source_sha256"] for row in bn_rows),
            "BN diagnostics have different training source hashes")
    pairs = []
    for first, second in (("default_bn", "momentum_only"),
                          ("epsilon_only", "both_bn_changes")):
        left, right = states[first], states[second]
        parameter_keys = [key for key in left if not key.endswith(BUFFER_SUFFIXES)]
        require(list(left) == list(right) and
                all(torch.equal(left[key], right[key]) for key in parameter_keys),
                f"{first}/{second}: momentum changed trained parameters")
        train_fields = ("train_loss", "train_acc", "gradient_norm_mean", "gradient_norm_max")
        require(all(all(a[key] == b[key] for key in train_fields)
                    for a, b in zip(histories[first], histories[second])),
                f"{first}/{second}: training trajectories changed")
        changed = [key for key in left if not torch.equal(left[key], right[key])]
        require(changed and all(key.endswith((".running_mean", ".running_var")) for key in changed),
                f"{first}/{second}: unexpected changed state tensors")
        pairs.append({"runs": [first, second], "trained_parameters_bitwise_equal": True,
                      "training_curves_bitwise_equal": True, "changed_bn_buffers": changed})
    return {"status": "PASS", "evaluation_scope": "A01 source-Session internal validation only",
            "target_session_evaluations": 0, "common_configuration": reference["config"],
            "data_sha256": reference["data_hash"], "fit_indices": reference["fit"],
            "validation_indices": reference["val"], "runs": rows,
            "momentum_pair_checks": pairs, "artifacts": artifacts}


def report_text(result):
    lines = ["# A01 implementation diagnostics audit", "",
             "PASS: five 1000-epoch runs; identical seed, initialization, data and source-Session split.",
             "230 fitting trials and 58 internal validation trials; zero target-Session evaluations.",
             "All results below are internal validation results, not paper-level test accuracy.", "",
             "| Variant | BN eps | BN momentum | Head dropout | Final val (%) | Best val (%) | Best epoch | Last 100 val mean (%) | Seconds |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["runs"]:
        lines.append(f"| {row['label']} | {row['bn_eps']:g} | {row['bn_momentum']:g} | "
                     f"{row['head_dropout']:g} | {100 * row['final_validation_accuracy']:.2f} | "
                     f"{100 * row['best_validation_accuracy']:.2f} | {row['best_epoch']} | "
                     f"{100 * row['late_validation_mean']:.2f} | {row['seconds']:.2f} |")
    lines.extend(["", "## Verified", "",
                  "- Clean commits match tags v1.0.26 and v1.0.27; all four BN runs share source hashes.",
                  "- The fit/validation indices are disjoint and jointly cover all 288 source trials.",
                  "- Prediction files contain validation labels/predictions and no target labels/predictions.",
                  "- Saved final weights reproduce the validation predictions and loss; repeated evaluation is identical and leaves BN state unchanged.",
                  "- All losses, gradient norms, weights and BN statistics are finite; BN variances are nonnegative.",
                  "- All 20 periodic checkpoints exist and agree with data, commit and split metadata.",
                  "- For each fixed epsilon, changing BN momentum leaves all learned parameters and training curves bitwise equal; only BN running means/variances change.",
                  "", "## Interpretation", "",
                  "At epoch 1000, the epsilon-only run differs from the default run by +10.34 percentage points on internal validation. This is not a small endpoint difference. Both momentum settings give the same endpoint accuracy at each epsilon, although their validation trajectories differ.",
                  "The head-dropout-off endpoint is 74.14%, compared with 67.24% for the default diagnostic; its best validation accuracy is lower (82.76% versus 87.93%). There is no consistent improvement across these summaries.",
                  "These are single-seed, single-subject diagnostic results, not proof of the authors' unpublished settings and not a nine-subject reproduction of 81.17%. No new nine-subject test run is justified solely by selecting the largest value here.",
                  "Existing target-Session results were inspected across successive candidates and remain exploratory even when each run evaluated the test Session only once. A future frozen-config run on those same subjects is not an untouched confirmatory test.",
                  "", "## Source and limitations", "",
                  "The paper specifies three branches, fusion positions, ECA, and a Flatten/Dense(32)/Dense(classes) head, but does not explicitly specify head dropout or BN numerical defaults. These variants therefore audit reconstruction assumptions; they are not asserted to be author code.",
                  "[AMEEGNet original article](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1540033/full)",
                  "The checkpoints save optimizer and RNG state but the runner has no resume command yet. Training history is stored in the final JSON, not continuously appended to JSONL. Neither capability is claimed by this audit.",
                  "", "## Validation confusion matrices", "",
                  "Rows are true classes; columns are predicted classes (left, right, feet, tongue)."])
    for row in result["runs"]:
        lines.extend(["", f"### {row['label']}", "", "```json",
                      json.dumps(row["validation_confusion_matrix"]), "```"])
    lines.extend(["", "## Artifact SHA-256", "", "| File | Bytes | SHA-256 |", "| --- | ---: | --- |"])
    for artifact in result["artifacts"]:
        name = Path(artifact["path"]).relative_to(Path(artifact["path"]).parents[1])
        lines.append(f"| {name.as_posix()} | {artifact['bytes']} | {artifact['sha256']} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("results"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    require(args.report.resolve() != args.manifest.resolve(), "Report and manifest must differ")
    require(not args.report.exists() and not args.manifest.exists(), "Refusing to overwrite audit outputs")
    result = audit(args.directory, torch.device(args.device))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.report.write_text(report_text(result), encoding="utf-8")
    print("PASS: 5 diagnostic runs, 20 checkpoints, 0 target evaluations")
    for row in result["runs"]:
        print(f"{row['label']}: final validation {100 * row['final_validation_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
