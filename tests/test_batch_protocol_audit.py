import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audit_hidden_maxnorm_batch import comparable_config


class BatchProtocolAuditTests(unittest.TestCase):
    def fixture(self, root, fit_samples=288, duplicate_index=False):
        result = {
            "epochs": 1000, "test_evaluations": 1,
            "validation_fraction": None, "final_test_acc": 1.0,
            "seconds": 1.0,
            "training_protocol": {
                "fit_samples": fit_samples, "validation_samples": None,
                "test_samples": 288,
            },
            "history": [{"epoch": i, "train_loss": 0.1, "test_acc": None}
                        for i in range(1, 1001)],
        }
        (root / "A01.json").write_text(json.dumps(result), encoding="utf-8")
        fit_indices = np.arange(fit_samples)
        if duplicate_index:
            fit_indices[-1] = 0
        labels = np.repeat(np.arange(4), 72)
        empty = np.array([], dtype=np.int64)
        np.savez(root / "A01.npz", y_true=labels, y_pred=labels,
                 fit_indices=fit_indices, validation_indices=empty,
                 validation_y_true=empty, validation_y_pred=empty)
        (root / "A01.pt").touch()

    def audit(self, root):
        result = subprocess.run(
            [sys.executable, "audit_hidden_maxnorm_batch.py", "--results", str(root),
             "--subjects", "A01", "--expected-fit-samples", "288"],
            capture_output=True, text=True, check=False,
        )
        return result.returncode, json.loads(result.stdout)

    def test_full_source_protocol_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            code, report = self.audit(root)
            self.assertEqual(code, 0, report["errors"])

    def test_screening_fit_cannot_pass_full_source_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, fit_samples=230)
            code, report = self.audit(root)
            self.assertEqual(code, 1)
            self.assertTrue(any("fit_samples=230, expected 288" in e for e in report["errors"]))

    def test_duplicate_fit_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, duplicate_index=True)
            code, report = self.audit(root)
            self.assertEqual(code, 1)
            self.assertTrue(any("cover full source" in e for e in report["errors"]))

    def test_source_split_is_a_comparable_setting(self):
        first = {"validation_fraction": None}
        second = {"validation_fraction": 0.2}
        self.assertNotEqual(comparable_config(first), comparable_config(second))

    def test_duplicate_epoch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            path = root / "A01.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            result["history"][-1]["epoch"] = 999
            path.write_text(json.dumps(result), encoding="utf-8")
            code, report = self.audit(root)
            self.assertEqual(code, 1)
            self.assertTrue(any("epoch sequence" in e for e in report["errors"]))


if __name__ == "__main__":
    unittest.main()
