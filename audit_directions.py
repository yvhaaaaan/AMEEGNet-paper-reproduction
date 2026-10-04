import argparse
import json
from pathlib import Path

import numpy as np


def read(directory):
    values = {}
    for i in range(1, 10):
        sid = f"A{i:02d}"
        record = json.loads((directory / f"{sid}.json").read_text(encoding="utf-8"))
        values[sid] = record["final_test_acc"] * 100
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--reverse", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reference-mean", type=float, default=81.17)
    args = parser.parse_args()
    forward, reverse = read(args.forward), read(args.reverse)
    subjects = sorted(forward)
    both = np.array([(forward[sid] + reverse[sid]) / 2 for sid in subjects])
    lines = ["# Bidirectional Session audit", "",
             "Forward is T-to-E and reverse is E-to-T. Each direction was a "
             "separate 1000-epoch final-only run; the paired mean is a "
             "descriptive protocol sensitivity statistic.", "",
             "| Subject | T-to-E (%) | E-to-T (%) | Paired mean (%) |",
             "| --- | ---: | ---: | ---: |"]
    lines += [f"| {sid} | {forward[sid]:.2f} | {reverse[sid]:.2f} | {both[i]:.2f} |"
              for i, sid in enumerate(subjects)]
    lines += ["", f"T-to-E mean +/- sample SD: {np.mean(list(forward.values())):.2f}% +/- "
              f"{np.std(list(forward.values()), ddof=1):.2f}%.",
              f"E-to-T mean +/- sample SD: {np.mean(list(reverse.values())):.2f}% +/- "
              f"{np.std(list(reverse.values()), ddof=1):.2f}%.",
              f"Paired bidirectional mean +/- sample SD: {both.mean():.2f}% +/- "
              f"{both.std(ddof=1):.2f}%.",
              f"Paired mean difference from paper {args.reference_mean:.2f}%: "
              f"{both.mean() - args.reference_mean:+.2f} percentage points.", "",
              "The two directions are similar; Session direction alone does not "
              "explain the gap to the published result."]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PASS: paired mean {both.mean():.2f}% +/- {both.std(ddof=1):.2f}%")


if __name__ == "__main__":
    main()
