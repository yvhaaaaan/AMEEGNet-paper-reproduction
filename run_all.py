import argparse, json, subprocess, sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--strict", action="store_true")
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
               "--device", a.device, "--out", str(path)]
        if a.strict:
            cmd.append("--strict")
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
