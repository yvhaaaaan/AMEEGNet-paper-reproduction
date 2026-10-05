import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path

# CUDA requires this to be set before CUDA work starts for deterministic GEMMs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ameegnet_ours import AMEEGNet, load_subject_npz, session_standardize


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_metadata():
    repository = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository,
            text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repository, text=True,
            stderr=subprocess.DEVNULL
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return commit, dirty


def gradient_norm(model):
    norms = [parameter.grad.detach().norm(2) for parameter in model.parameters()
             if parameter.grad is not None]
    value = torch.stack(norms).norm(2)
    if not torch.isfinite(value):
        raise FloatingPointError("Non-finite gradient norm")
    return float(value.item())


def configure_reproducibility(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def evaluate(model, x, y, device, loss_fn):
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(x).to(device)
        yt = torch.from_numpy(y).to(device)
        logits = model(xt)
        loss = float(loss_fn(logits, yt).item())
        pred = logits.argmax(1).cpu().numpy()
    return loss, float(accuracy_score(y, pred)), pred


def normalize_trials(x, mode):
    if mode == "none":
        return x
    axes = (1, 2) if mode == "trial" else (2,)
    mean = x.mean(axis=axes, keepdims=True)
    std = x.std(axis=axes, keepdims=True).clip(min=1e-6)
    return ((x - mean) / std).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--input-scale", type=float, default=1.0)
    p.add_argument("--input-normalization", choices=("none", "trial", "channel-trial"),
                   default="none")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--deterministic", action="store_true",
                   help="enable deterministic PyTorch/CUDA algorithms")
    p.add_argument("--validation-fraction", type=float, default=None,
                   help="split this fraction from the training session for diagnosis")
    p.add_argument("--test-evaluation", choices=("final", "none", "each-epoch"),
                   default="final",
                   help="when to evaluate the held-out target session")
    p.add_argument("--no-bn-first", action="store_true")
    p.add_argument("--bn-eps", type=float, default=1e-5)
    p.add_argument("--bn-momentum", type=float, default=0.1,
                   help="PyTorch weight of the new batch statistics, not Keras momentum")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--elu-before-bn", action="store_true")
    p.add_argument("--paper-pooling", action="store_true")
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--dropout-after-pool", action="store_true")
    p.add_argument("--no-head-dropout", action="store_true")
    p.add_argument("--max-norm", action="store_true")
    p.add_argument("--hidden-max-norm", action="store_true")
    p.add_argument("--no-head-elu", action="store_true")
    p.add_argument("--reverse-sessions", action="store_true")
    p.add_argument("--no-fusion", action="store_true")
    p.add_argument("--no-eca", action="store_true")
    p.add_argument("--eca-bias", action="store_true")
    p.add_argument("--out", default="results/s01.json")
    a = p.parse_args()
    out = Path(a.out)
    if a.epochs <= 0 or a.log_every < 0 or a.checkpoint_every < 0:
        raise ValueError("epochs must be positive and log/checkpoint intervals nonnegative")
    if not math.isfinite(a.input_scale) or a.input_scale <= 0:
        raise ValueError("input-scale must be finite and positive")
    if a.dropout is not None and (not math.isfinite(a.dropout) or not 0 <= a.dropout < 1):
        raise ValueError("dropout must be finite and in [0,1)")
    checkpoint_dir = out.parent / (out.stem + "_checkpoints")
    artifacts = [out, out.with_suffix('.pt'), out.with_suffix('.npz')]
    if any(path.exists() for path in artifacts):
        raise FileExistsError(f"Refusing to overwrite experiment artifacts: {out}")
    if a.checkpoint_every and checkpoint_dir.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoints: {checkpoint_dir}")
    commit, dirty = git_metadata()
    repository = Path(__file__).resolve().parent
    version = (repository / "VERSION").read_text(encoding="utf-8").strip()
    data_digest = file_sha256(a.data)
    source_hashes = {name: file_sha256(repository / name) for name in
                     ("run_s01.py", "ameegnet_ours/model.py", "ameegnet_ours/data.py")}

    configure_reproducibility(a.seed, a.deterministic)
    device = torch.device(a.device if a.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    xtr, ytr, xte, yte = load_subject_npz(a.data)
    xtr = (xtr * a.input_scale).astype(np.float32)
    xte = (xte * a.input_scale).astype(np.float32)
    xtr = normalize_trials(xtr, a.input_normalization)
    xte = normalize_trials(xte, a.input_normalization)
    if a.reverse_sessions:
        xtr, ytr, xte, yte = xte, yte, xtr, ytr
    if not a.strict:
        xtr, xte = session_standardize(xtr, xte)

    validation_fraction = a.validation_fraction
    if validation_fraction is None and not a.strict:
        validation_fraction = 0.15
    if validation_fraction is not None and not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must be between 0 and 1")
    if validation_fraction is None:
        fit, val = np.arange(len(ytr)), None
    else:
        fit, val = train_test_split(
            np.arange(len(ytr)), test_size=validation_fraction,
            stratify=ytr, random_state=a.seed
        )
        fit, val = np.asarray(fit), np.asarray(val)

    use_pool = a.paper_pooling or not a.strict
    dropout = a.dropout if a.dropout is not None else (
        0.25 if (a.paper_pooling or not a.strict) else 0.0
    )
    head_dropout = 0.0 if a.no_head_dropout else dropout
    model = AMEEGNet(
        pool=use_pool, dropout=dropout,
        bn_first=not a.no_bn_first,
        norm_then_activation=not a.elu_before_bn,
        fusion=not a.no_fusion, eca=not a.no_eca,
        dropout_after_pool=a.dropout_after_pool,
        head_elu=not a.no_head_elu,
        head_dropout=head_dropout,
        bn_eps=a.bn_eps, bn_momentum=a.bn_momentum,
        eca_bias=a.eca_bias,
    ).to(device)
    initial_fingerprint = state_fingerprint(model)
    opt = (torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
           if a.strict else torch.optim.AdamW(model.parameters(), lr=1e-3,
                                              weight_decay=1e-4))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1 if not a.strict else 0.0)
    gen = torch.Generator().manual_seed(a.seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(xtr[fit]), torch.from_numpy(ytr[fit])),
        batch_size=64, shuffle=True, generator=gen
    )

    best = (-1.0, None, 0)
    history = []
    test_evaluations = 0
    t0 = time.perf_counter()
    if a.checkpoint_every:
        checkpoint_dir.mkdir(parents=True)
    for ep in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        preds, ys = [], []
        grad_norms = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {ep}")
            loss.backward()
            grad_norms.append(gradient_norm(model))
            if not a.strict:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if a.max_norm:
                model.project_eegnet_max_norm(hidden_max=0.25 if a.hidden_max_norm else None)
            total += loss.item() * len(yb)
            preds.extend(logits.argmax(1).detach().cpu().numpy())
            ys.extend(yb.cpu().numpy())

        train_loss = total / len(fit)
        train_acc = float(accuracy_score(ys, preds))
        val_loss = val_acc = None
        if val is not None:
            val_loss, val_acc, _ = evaluate(model, xtr[val], ytr[val], device, loss_fn)
            if val_acc > best[0]:
                best = (val_acc,
                        {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        ep)
        test_acc = None
        if a.test_evaluation == "each-epoch":
            _, test_acc, _ = evaluate(model, xte, yte, device, loss_fn)
            test_evaluations += 1
        history.append({"epoch": ep, "train_loss": train_loss,
                        "train_acc": train_acc, "val_loss": val_loss,
                        "val_acc": val_acc, "test_acc": test_acc,
                        "gradient_norm_mean": float(np.mean(grad_norms)),
                        "gradient_norm_max": float(np.max(grad_norms))})
        if a.log_every and (ep % a.log_every == 0 or ep == a.epochs):
            print(json.dumps(history[-1]), flush=True)
        if a.checkpoint_every and (ep % a.checkpoint_every == 0 or ep == a.epochs):
            checkpoint = checkpoint_dir / f"epoch_{ep:04d}.pt"
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "epoch": ep, "args": vars(a), "git_commit": commit,
                        "data_sha256": data_digest, "loader_rng": gen.get_state(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                        "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                        "fit_indices": fit, "validation_indices": val}, checkpoint)

    if best[1] is not None and not a.strict:
        model.load_state_dict(best[1])
    final_val_loss = final_val_acc = None
    val_pred = np.array([], dtype=np.int64)
    if val is not None:
        final_val_loss, final_val_acc, val_pred = evaluate(
            model, xtr[val], ytr[val], device, loss_fn
        )
    final_test_acc = None
    final_pred = np.array([], dtype=np.int64)
    if a.test_evaluation != "none":
        _, final_test_acc, final_pred = evaluate(model, xte, yte, device, loss_fn)
        test_evaluations += 1
    elapsed = time.perf_counter() - t0
    result = {
        "version": version,
        "git_commit": commit, "git_dirty": dirty,
        "data_path": str(Path(a.data).resolve()),
        "data_sha256": data_digest, "source_sha256": source_hashes,
        "input_scale": a.input_scale,
        "input_normalization": a.input_normalization,
        "strict": a.strict, "paper_pooling": a.paper_pooling,
        "dropout": dropout, "dropout_after_pool": a.dropout_after_pool,
        "head_dropout": head_dropout,
        "max_norm": a.max_norm, "head_elu": not a.no_head_elu,
        "hidden_max_norm": a.hidden_max_norm,
        "reverse_sessions": a.reverse_sessions, "fusion": not a.no_fusion,
        "eca": not a.no_eca, "bn_first": not a.no_bn_first,
        "eca_bias": a.eca_bias,
        "norm_then_activation": not a.elu_before_bn,
        "bn_eps": a.bn_eps, "bn_momentum": a.bn_momentum,
        "validation_fraction": validation_fraction,
        "seed": a.seed, "epochs": a.epochs, "device": str(device),
        "deterministic": a.deterministic,
        "test_evaluation": a.test_evaluation,
        "final_test_acc": final_test_acc,
        "final_val_acc": final_val_acc, "final_val_loss": final_val_loss,
        "checkpoint_selection": "final_epoch" if a.strict or val is None else "best_internal_val_accuracy",
        "best_val_acc": None if val is None else best[0],
        "best_epoch": None if val is None else best[2],
        "seconds": elapsed, "test_evaluations": test_evaluations,
        "initial_state_sha256": initial_fingerprint,
        "final_state_sha256": state_fingerprint(model),
        "history": history,
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        },
    }
    result["training_protocol"] = {
        "optimizer": type(opt).__name__, "lr": 1e-3,
        "weight_decay": opt.defaults["weight_decay"], "betas": opt.defaults["betas"],
        "eps": opt.defaults["eps"],
        "gradient_clip_max_norm": None if a.strict else 5.0,
        "batch_size": 64, "fit_samples": len(fit),
        "validation_samples": None if val is None else len(val),
        "test_samples": len(yte), "label_smoothing": 0.0 if a.strict else 0.1,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "args": vars(a), "result": result}, out.with_suffix(".pt"))
    np.savez(out.with_suffix(".npz"),
             y_true=yte if final_pred.size else np.array([], dtype=np.int64),
             y_pred=final_pred, fit_indices=fit,
             validation_indices=np.array([], dtype=np.int64) if val is None else val,
             validation_y_true=np.array([], dtype=np.int64) if val is None else ytr[val],
             validation_y_pred=val_pred)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
