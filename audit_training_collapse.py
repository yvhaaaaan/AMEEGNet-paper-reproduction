"""Inspect frozen checkpoints on the source session without target evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from audit_source_bn import construct
from run_s01 import file_sha256, normalize_trials, state_fingerprint


def activation_stats(values):
    values = values.double()
    std = values.std(dim=0, unbiased=False)
    return {
        "shape": list(values.shape),
        "finite": bool(torch.isfinite(values).all()),
        "minimum": float(values.min()), "maximum": float(values.max()),
        "mean": float(values.mean()),
        "per_unit_mean": values.mean(dim=0).tolist(),
        "per_unit_std": std.tolist(),
        "units_with_std_below_1e_5": int((std < 1e-5).sum()),
        "fraction_below_minus_10": float((values < -10).double().mean()),
        "fraction_below_minus_20": float((values < -20).double().mean()),
    }


@torch.inference_mode()
def inspect_source(model, inputs, labels, batch_size=32):
    model.eval()
    before = state_fingerprint(model)
    hidden_pre, hidden_post, outputs = [], [], []

    def capture(destination):
        def hook(module, args, output):
            destination.append(output.detach().cpu())
        return hook

    handles = [model.head[0].register_forward_hook(capture(hidden_pre))]
    if isinstance(model.head[1], nn.ELU):
        handles.append(model.head[1].register_forward_hook(capture(hidden_post)))
    try:
        for start in range(0, len(inputs), batch_size):
            outputs.append(model(torch.from_numpy(inputs[start:start + batch_size])).cpu())
    finally:
        for handle in handles:
            handle.remove()
    if state_fingerprint(model) != before:
        raise AssertionError("diagnosis changed model parameters or buffers")
    logits = torch.cat(outputs)
    if not torch.isfinite(logits).all():
        raise FloatingPointError("non-finite source logits")
    target = torch.from_numpy(labels)
    prediction = logits.argmax(dim=1)
    return {
        "source_samples": len(labels),
        "source_accuracy": float((prediction == target).float().mean()),
        "source_loss": float(nn.functional.cross_entropy(logits, target)),
        "source_prediction_counts": torch.bincount(prediction, minlength=4).tolist(),
        "source_true_counts": torch.bincount(target, minlength=4).tolist(),
        "hidden_pre_elu": activation_stats(torch.cat(hidden_pre)),
        "hidden_post_elu": activation_stats(torch.cat(hidden_post)) if hidden_post else None,
        "logits": activation_stats(logits),
        "unchanged_state_sha256": before,
        "bn_buffers": {
            name: {
                "running_mean_abs_max": float(layer.running_mean.abs().max()),
                "running_var_min": float(layer.running_var.min()),
                "running_var_max": float(layer.running_var.max()),
                "gamma_abs_max": float(layer.weight.abs().max()),
                "beta_abs_max": float(layer.bias.abs().max()),
            }
            for name, layer in model.named_modules() if isinstance(layer, nn.BatchNorm2d)
        },
    }


def audit_checkpoint(checkpoint, batch_size):
    checkpoint = Path(checkpoint)
    digest = file_sha256(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu")
    config = saved["result"]
    if not config["strict"] or config["reverse_sessions"]:
        raise ValueError("diagnosis expects the strict T -> E configuration")
    if config.get("validation_fraction") is not None:
        raise ValueError("use a full-source checkpoint, not an internal split")
    data_path = Path(config["data_path"])
    if file_sha256(data_path) != config["data_sha256"]:
        raise AssertionError("data hash differs from the training input")
    with np.load(data_path, allow_pickle=False) as data:
        source = data["sessions"].astype(str) == "0train"
        x = data["x"][source].astype(np.float32)
        y = data["y"][source].astype(np.int64)
    if len(y) != config["training_protocol"]["fit_samples"]:
        raise AssertionError("source sample count differs")
    with np.load(checkpoint.with_suffix(".npz"), allow_pickle=False) as prediction:
        if not np.array_equal(np.sort(prediction["fit_indices"]), np.arange(len(y))):
            raise AssertionError("fit indices do not cover the source session")
        if prediction["validation_indices"].size:
            raise AssertionError("unexpected validation split")
    x = normalize_trials(x * config["input_scale"], config["input_normalization"])
    model = construct(config)
    model.load_state_dict(saved["model"], strict=True)
    if state_fingerprint(model) != config["final_state_sha256"]:
        raise AssertionError("checkpoint does not match the recorded final model")
    diagnostic = inspect_source(model, x, y, batch_size)
    if file_sha256(checkpoint) != digest:
        raise AssertionError("checkpoint file changed during diagnosis")
    return {
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": digest,
        "data_sha256": config["data_sha256"], "seed": config["seed"],
        "training_commit": config["git_commit"],
        "last_online_training_record": config["history"][-1],
        **diagnostic,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    torch.set_num_threads(1)
    report = {
        "scope": "source_only_frozen_checkpoint_diagnostic_not_a_paper_result",
        "device": "cpu", "target_evaluations": 0, "parameter_updates": 0,
        "bn_recalibration": False,
        "rows": [audit_checkpoint(path, args.batch_size) for path in args.checkpoint],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(args.out.resolve()), "target_evaluations": 0,
        "rows": [{key: row[key] for key in (
            "checkpoint", "seed", "source_accuracy", "source_loss",
            "source_prediction_counts",
        )} for row in report["rows"]],
    }, indent=2))


if __name__ == "__main__":
    main()
