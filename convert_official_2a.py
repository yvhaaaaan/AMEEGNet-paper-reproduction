import sys
from pathlib import Path
import numpy as np

v17 = Path(r"E:\论文\AMEEGNet_v1.7.0_review_20261003\AMEEGNet")
sys.path.insert(0, str(v17))
from ameegnet.data import load_within_subject

source = r"E:\论文\data\MNE-bnci-data\database\data-sets\001-2014"
out = Path("data")
out.mkdir(parents=True, exist_ok=True)
subjects = [f"A{i:02d}" for i in range(1, 10)]
records = load_within_subject("2a", source, subjects)
for sid, d in records.items():
    x = np.concatenate([d["X_train"], d["X_test"]]).astype(np.float32)
    y = np.concatenate([d["y_train"], d["y_test"]]).astype(np.int64)
    sessions = np.array(["0train"] * len(d["y_train"]) + ["1test"] * len(d["y_test"]))
    np.savez_compressed(out / f"{sid}.npz", x=x, y=y, sessions=sessions)
    print(sid, x.shape, np.bincount(y, minlength=4).tolist())
