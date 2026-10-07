"""Compare cuDNN precision permissions on frozen source-validation weights."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from audit_source_bn import construct
from audit_source_head_validation import load_run
from run_s01 import (
    configure_reproducibility, file_sha256, git_metadata, normalize_trials,
    state_fingerprint,
)


def compare_logits(fp32, tf32_allowed, labels):
    if fp32.ndim != 2 or fp32.shape != tf32_allowed.shape or fp32.shape[1] != 4:
        raise ValueError("expected equal (N,4) logits")
    if labels.shape != (fp32.shape[0],) or not labels.numel():
        raise ValueError("incorrect or empty source labels")
    if labels.dtype != torch.int64 or not ((labels >= 0) & (labels < 4)).all():
        raise ValueError("source labels must be integer class indices")
    if not torch.isfinite(fp32).all() or not torch.isfinite(tf32_allowed).all():
        raise FloatingPointError("non-finite source logits")
    fp_pred = fp32.argmax(dim=1)
    tf_pred = tf32_allowed.argmax(dim=1)
    difference = (tf32_allowed - fp32).double()
    return {
        "source_validation_samples": labels.numel(),
        "fp32_accuracy": float((fp_pred == labels).double().mean()),
        "tf32_allowed_accuracy": float((tf_pred == labels).double().mean()),
        "fp32_loss": float(nn.functional.cross_entropy(fp32, labels)),
        "tf32_allowed_loss": float(nn.functional.cross_entropy(tf32_allowed, labels)),
        "logits_exactly_equal": torch.equal(fp32, tf32_allowed),
        "max_abs_logit_difference": float(difference.abs().max()),
        "rms_logit_difference": float(difference.square().mean().sqrt()),
        "changed_prediction_trials": int((fp_pred != tf_pred).sum()),
    }


@torch.inference_mode()
def precision_pair(model, inputs):
    if model.training:
        raise ValueError("frozen diagnosis requires eval mode")
    before = state_fingerprint(model)
    previous_tf32 = torch.backends.cudnn.allow_tf32
    logits = []
    try:
        for allowed in (False, True):
            torch.backends.cudnn.allow_tf32 = allowed
            logits.append(model(inputs).detach().cpu())
    finally:
        torch.backends.cudnn.allow_tf32 = previous_tf32
    if state_fingerprint(model) != before:
        raise AssertionError("precision diagnosis changed weights or BN buffers")
    return logits


def audit_checkpoint(checkpoint):
    checkpoint = Path(checkpoint)
    before_sha = file_sha256(checkpoint)
    config, predictions, errors = load_run(checkpoint.with_suffix(".json"), True,
                                         json.loads(checkpoint.with_suffix(".json").read_text(
                                             encoding="utf-8"))["seed"])
    if errors:
        raise ValueError(errors)
    if not config["strict"] or config["reverse_sessions"]:
        raise ValueError("expected strict T-session source checkpoint")
    runtime = config["runtime"]
    required_flags = {
        "deterministic_algorithms": True, "cudnn_deterministic": True,
        "cudnn_benchmark": False, "matmul_tf32": False, "cudnn_tf32": False,
    }
    if any(runtime.get(key) != value for key, value in required_flags.items()):
        raise ValueError("reference backend flags differ from registration")
    repository = Path(__file__).resolve().parent
    expected_sources = {"run_s01.py", "ameegnet_ours/model.py", "ameegnet_ours/data.py"}
    if set(config["source_sha256"]) != expected_sources:
        raise ValueError("missing training source hashes")
    for name, expected in config["source_sha256"].items():
        if file_sha256(repository / name) != expected:
            raise AssertionError(f"training source changed: {name}")
    data_path = Path(config["data_path"])
    if file_sha256(data_path) != config["data_sha256"]:
        raise AssertionError("source cache changed")
    with np.load(data_path, allow_pickle=False) as data:
        source = data["sessions"].astype(str) == "0train"
        x = data["x"][source].astype(np.float32)
        y = data["y"][source].astype(np.int64)
    if x.shape != (288, 22, 1125) or y.shape != (288,):
        raise ValueError("incorrect source data shape")
    indices = predictions["validation_indices"]
    if not np.array_equal(y[indices], predictions["validation_y_true"]):
        raise AssertionError("validation labels differ from source cache")
    x = normalize_trials(x[indices] * config["input_scale"], config["input_normalization"])
    saved = torch.load(checkpoint, map_location="cpu")
    if saved["result"] != config:
        raise AssertionError("PT and JSON results differ")
    configure_reproducibility(config["seed"], True)
    model = construct(config).to("cuda").eval()
    model.load_state_dict(saved["model"], strict=True)
    final_fingerprint = state_fingerprint(model)
    if final_fingerprint != config["final_state_sha256"]:
        raise AssertionError("not the recorded final checkpoint")
    fp32, tf32_allowed = precision_pair(model, torch.from_numpy(x).to("cuda"))
    if not np.array_equal(fp32.argmax(dim=1).numpy(), predictions["validation_y_pred"]):
        raise AssertionError("FP32 baseline does not reproduce source validation predictions")
    labels = torch.from_numpy(y[indices])
    report = compare_logits(fp32, tf32_allowed, labels)
    if file_sha256(checkpoint) != before_sha:
        raise AssertionError("checkpoint changed during diagnosis")
    return {
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": before_sha,
        "data_sha256": config["data_sha256"], "seed": config["seed"],
        "training_commit": config["git_commit"],
        "unchanged_state_sha256": final_fingerprint,
        **report,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if len(set(path.resolve() for path in args.checkpoint)) != len(args.checkpoint):
        raise ValueError("duplicate reference checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("this backend diagnostic requires CUDA")
    torch.set_num_threads(1)
    commit, dirty = git_metadata()
    repository = Path(__file__).resolve().parent
    report = {
        "scope": "frozen_source_precision_diagnostic_not_training_or_paper_accuracy",
        "device": "cuda", "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "target_evaluations": 0, "parameter_updates": 0, "bn_recalibration": False,
        "changed_factor": "cudnn.allow_tf32 permission only",
        "matmul_tf32": False, "deterministic_algorithms": True,
        "cudnn_deterministic": True, "cudnn_benchmark": False,
        "diagnostic_git_commit": commit, "diagnostic_git_dirty": dirty,
        "diagnostic_source_sha256": {name: file_sha256(repository / name) for name in (
            "audit_frozen_tf32.py", "audit_source_bn.py", "audit_source_head_validation.py",
            "run_s01.py", "ameegnet_ours/model.py", "ameegnet_ours/data.py",
        )},
        "rows": [audit_checkpoint(path) for path in args.checkpoint],
    }
    report["n_source_checkpoints"] = len(report["rows"])
    report["any_logit_difference"] = any(not row["logits_exactly_equal"] for row in report["rows"])
    report["total_changed_source_prediction_trials"] = sum(
        row["changed_prediction_trials"] for row in report["rows"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
