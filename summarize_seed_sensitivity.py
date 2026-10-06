import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from audit_hidden_maxnorm_batch import comparable_config


SEEDS = (1, 7, 2025)
SUBJECTS = tuple(f"A{i:02d}" for i in range(1, 10))


def summarize(root):
    accuracies = np.empty((len(SUBJECTS), len(SEEDS)), dtype=float)
    rows = []
    errors = []
    runs = []
    seed_reports = []
    for column, seed in enumerate(SEEDS):
        batch = root / f"seed{seed}"
        audit = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("audit_hidden_maxnorm_batch.py")),
             "--results", str(batch), "--expected-fit-samples", "288"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if not audit.stdout.strip():
            raise RuntimeError(f"seed {seed}: audit did not produce JSON: {audit.stderr}")
        report = json.loads(audit.stdout)
        errors.extend(f"seed {seed}: {item}" for item in report["errors"])
        if audit.returncode and not report["errors"]:
            errors.append(f"seed {seed}: audit exit code {audit.returncode}")
        seed_reports.append({"seed": seed, "audit": report})
        for row, subject in enumerate(SUBJECTS):
            result = json.loads((batch / f"{subject}.json").read_text(encoding="utf-8"))
            if result.get("seed") != seed:
                errors.append(f"{subject}/seed {seed}: wrong saved seed")
            if result.get("validation_fraction") is not None:
                errors.append(f"{subject}/seed {seed}: unexpected source split")
            accuracies[row, column] = result["final_test_acc"]
            runs.append((subject, seed, result))

    reference = comparable_config(runs[0][2])
    reference.pop("seed")
    reference_hashes = runs[0][2]["source_sha256"]
    data_hashes = {}
    for subject, seed, result in runs:
        config = comparable_config(result)
        config.pop("seed")
        if config != reference:
            errors.append(f"{subject}/seed {seed}: configuration differs beyond seed")
        if result["source_sha256"] != reference_hashes:
            errors.append(f"{subject}/seed {seed}: source hashes differ")
        data_hashes.setdefault(subject, result["data_sha256"])
        if result["data_sha256"] != data_hashes[subject]:
            errors.append(f"{subject}/seed {seed}: input hash differs")

    subject_means = accuracies.mean(axis=1)
    artifacts = []
    for subject, seed, _ in runs:
        for extension in ("json", "npz", "pt"):
            path = root / f"seed{seed}" / f"{subject}.{extension}"
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            size = path.stat().st_size
            if not size:
                errors.append(f"{subject}/seed {seed}: empty {extension} artifact")
            artifacts.append({
                "path": str(path.relative_to(root)), "bytes": size,
                "sha256": digest.hexdigest(),
            })
    for row, subject in enumerate(SUBJECTS):
        rows.append({
            "subject": subject,
            **{f"seed_{seed}": float(accuracies[row, column])
               for column, seed in enumerate(SEEDS)},
            "seed_mean": float(subject_means[row]),
            "seed_sample_sd": float(accuracies[row].std(ddof=1)),
        })
    seed_means = accuracies.mean(axis=0)
    return {
        "results": str(root.resolve()), "seeds": SEEDS, "n_subjects": len(SUBJECTS),
        "n_runs": len(runs), "subjects": rows, "seed_reports": seed_reports,
        "artifact_manifest": artifacts,
        "mean_subject_seed_average": float(subject_means.mean()),
        "sample_sd_subject_seed_average": float(subject_means.std(ddof=1)),
        "sample_sd_seed_means": float(seed_means.std(ddof=1)),
        "pooled_run_sample_sd_descriptive_only": float(accuracies.std(ddof=1)),
        "pooled_runs_are_independent_subjects": False,
        "paper_mean": 0.8117,
        "difference_from_paper_pp": float((subject_means.mean() - 0.8117) * 100),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    csv_path = out.with_suffix(".csv")
    if out.exists() or csv_path.exists():
        raise FileExistsError(f"Refusing to overwrite summary: {out}")
    report = summarize(Path(args.results))
    if report["errors"]:
        print(json.dumps(report, indent=2, ensure_ascii=True))
        raise SystemExit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["subjects"][0]))
        writer.writeheader()
        writer.writerows(report["subjects"])
    print(json.dumps({
        key: value for key, value in report.items()
        if key not in ("seed_reports", "artifact_manifest")
    }, indent=2, ensure_ascii=True))
    print(json.dumps({"seed_results": [{
        "seed": item["seed"], "mean": item["audit"]["mean_accuracy"],
        "sample_sd": item["audit"]["sample_std_accuracy"],
        "seconds": item["audit"]["runtime_total_seconds"],
    } for item in report["seed_reports"]]}, indent=2))


if __name__ == "__main__":
    main()
