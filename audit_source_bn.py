"""Diagnose saved BatchNorm buffers using source-Session data only."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ameegnet_ours import AMEEGNet
from run_s01 import (configure_reproducibility, evaluate, file_sha256,
                     normalize_trials, state_fingerprint)


def construct(config):
    return AMEEGNet(
        pool=config["paper_pooling"] or not config["strict"],
        dropout=config["dropout"], fusion=config["fusion"], eca=config["eca"],
        bn_first=config["bn_first"],
        norm_then_activation=config["norm_then_activation"],
        dropout_after_pool=config["dropout_after_pool"],
        head_elu=config["head_elu"], head_dropout=config["head_dropout"],
        bn_eps=config["bn_eps"], bn_momentum=config["bn_momentum"],
        eca_bias=config.get("eca_bias", False),
        init_mode=config.get("init_mode", "default"),
        eca_stage=config.get("eca_stage", "output"),
        fixed_fusion_channels=config.get("fixed_fusion_channels", False),
    )


@torch.no_grad()
def recalibrate(model, inputs, mode):
    model.eval()
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    if mode == "single_source_batch":
        for bn in bns:
            bn.reset_running_stats()
            bn.momentum = 1.0
            bn.train()
        model(inputs)
    elif mode == "sequential_source_moments":
        # Recompute one layer at a time so downstream statistics see the
        # upstream layers in their final inference state, with Dropout off.
        for bn in bns:
            captured = []

            def capture(module, args):
                captured.append(args[0].detach())

            handle = bn.register_forward_pre_hook(capture)
            try:
                model(inputs)
            finally:
                handle.remove()
            if len(captured) != 1:
                raise AssertionError("expected one input per BatchNorm layer")
            values = captured[0].double()
            axes = (0, 2, 3)
            mean = values.mean(dim=axes)
            centered = values - mean.view(1, -1, 1, 1)
            count = values.numel() // values.shape[1]
            variance = centered.square().sum(dim=axes) / (count - 1)
            bn.running_mean.copy_(mean)
            bn.running_var.copy_(variance)
    else:
        raise ValueError(mode)
    model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    saved = torch.load(args.checkpoint, map_location="cpu")
    config = saved["result"]
    if config["test_evaluation"] != "none" or config["test_evaluations"] != 0:
        raise ValueError("use an internal-validation-only checkpoint")
    configure_reproducibility(config["seed"], True)
    device = torch.device(args.device)
    with np.load(config["data_path"], allow_pickle=False) as data:
        source = data["sessions"].astype(str) == "0train"
        x = data["x"][source].astype(np.float32)
        y = data["y"][source].astype(np.int64)
    if config["reverse_sessions"] or not config["strict"]:
        raise ValueError("diagnostic expects a strict T-session checkpoint")
    x = normalize_trials(x * config.get("input_scale", 1.0),
                         config.get("input_normalization", "none"))
    with np.load(args.checkpoint.with_suffix(".npz"), allow_pickle=False) as pred:
        fit = pred["fit_indices"]
        val = pred["validation_indices"]
        original_predictions = pred["validation_y_pred"]
        if pred["y_true"].size or pred["y_pred"].size:
            raise AssertionError("checkpoint has target predictions")
    if set(fit) & set(val) or len(fit) + len(val) != len(y) or not len(val):
        raise AssertionError("invalid source split")
    model = construct(config).to(device)
    model.load_state_dict(saved["model"], strict=True)
    loss_fn = nn.CrossEntropyLoss()
    baseline = evaluate(model, x[val], y[val], device, loss_fn)
    if not np.array_equal(baseline[2], original_predictions):
        raise AssertionError("saved source predictions not reproduced")
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    rows = [{"mode": "saved_buffers", "val_loss": baseline[0],
             "val_accuracy": baseline[1]}]
    for mode in ("single_source_batch", "sequential_source_moments"):
        alternative = copy.deepcopy(model)
        recalibrate(alternative, torch.from_numpy(x[fit]).to(device), mode)
        if any(not torch.equal(p, before[name])
               for name, p in alternative.named_parameters()):
            raise AssertionError("calibration changed a trainable parameter")
        loss, acc, prediction = evaluate(alternative, x[val], y[val], device, loss_fn)
        rows.append({"mode": mode, "val_loss": loss, "val_accuracy": acc,
                     "predictions_changed": int((prediction != baseline[2]).sum()),
                     "state_sha256": state_fingerprint(alternative)})
    report = {"scope": "source_only_BN_diagnostic_not_a_paper_result",
              "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": file_sha256(args.checkpoint),
              "fit_samples": len(fit), "validation_samples": len(val),
              "target_evaluations": 0, "parameter_updates": 0, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
