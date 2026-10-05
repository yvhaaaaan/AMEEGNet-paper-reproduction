import argparse, json, subprocess, sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--input-scale", type=float, default=1.0)
    p.add_argument("--input-normalization", choices=("none", "trial", "channel-trial"),
                   default="none")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--test-evaluation", choices=("final", "none", "each-epoch"),
                   default="final")
    p.add_argument("--validation-fraction", type=float, default=None)
    p.add_argument("--paper-pooling", action="store_true")
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--dropout-after-pool", action="store_true")
    p.add_argument("--no-head-dropout", action="store_true")
    p.add_argument("--max-norm", action="store_true")
    p.add_argument("--hidden-max-norm", action="store_true")
    p.add_argument("--no-bn-first", action="store_true")
    p.add_argument("--elu-before-bn", action="store_true")
    p.add_argument("--no-head-elu", action="store_true")
    p.add_argument("--reverse-sessions", action="store_true")
    p.add_argument("--no-fusion", action="store_true")
    p.add_argument("--no-eca", action="store_true")
    p.add_argument("--eca-bias", action="store_true")
    p.add_argument("--eca-stage", choices=("output", "depth_pre_sep"),
                   default="output")
    p.add_argument("--init-mode", choices=("default", "xavier_uniform",
                                             "xavier_normal", "kaiming_normal"),
                   default="default")
    p.add_argument("--bn-eps", type=float, default=1e-5)
    p.add_argument("--bn-momentum", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--out", default="results/all_500")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(1, 10):
        sid = f"A{i:02d}"
        path = out / f"{sid}.json"
        cmd = [sys.executable, "run_s01.py", "--data", f"data/{sid}.npz",
               "--epochs", str(a.epochs), "--seed", str(a.seed),
               "--batch-size", str(a.batch_size),
               "--device", a.device, "--input-scale", str(a.input_scale),
               "--input-normalization", a.input_normalization,
               "--out", str(path)]
        if a.strict:
            cmd.append("--strict")
        if a.deterministic:
            cmd.append("--deterministic")
        cmd.extend(["--test-evaluation", a.test_evaluation])
        if a.validation_fraction is not None:
            cmd.extend(["--validation-fraction", str(a.validation_fraction)])
        if a.paper_pooling:
            cmd.append("--paper-pooling")
        if a.dropout is not None:
            cmd.extend(["--dropout", str(a.dropout)])
        if a.dropout_after_pool:
            cmd.append("--dropout-after-pool")
        if a.no_head_dropout:
            cmd.append("--no-head-dropout")
        if a.max_norm:
            cmd.append("--max-norm")
        if a.hidden_max_norm:
            cmd.append("--hidden-max-norm")
        if a.no_bn_first:
            cmd.append("--no-bn-first")
        if a.elu_before_bn:
            cmd.append("--elu-before-bn")
        if a.no_head_elu:
            cmd.append("--no-head-elu")
        if a.reverse_sessions:
            cmd.append("--reverse-sessions")
        if a.no_fusion:
            cmd.append("--no-fusion")
        if a.no_eca:
            cmd.append("--no-eca")
        if a.eca_bias:
            cmd.append("--eca-bias")
        if a.eca_stage != "output":
            cmd.extend(["--eca-stage", a.eca_stage])
        if a.init_mode != "default":
            cmd.extend(["--init-mode", a.init_mode])
        cmd.extend(["--bn-eps", str(a.bn_eps), "--bn-momentum", str(a.bn_momentum),
                    "--log-every", str(a.log_every),
                    "--checkpoint-every", str(a.checkpoint_every)])
        print("running", sid, flush=True)
        subprocess.run(cmd, check=True)
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"subject": sid, **{k: result[k] for k in
                     ("final_test_acc", "best_val_acc", "best_epoch", "seconds")}})
        (out / "per_subject.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    acc = [r["final_test_acc"] for r in rows]
    mean = sum(acc) / len(acc)
    summary = {"subjects": rows, "mean_final_acc": mean,
               "std_final_acc": (sum((x - mean) ** 2 for x in acc) / (len(acc) - 1)) ** 0.5}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
