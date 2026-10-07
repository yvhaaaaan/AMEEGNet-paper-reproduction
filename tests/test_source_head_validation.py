import copy
import unittest

import numpy as np

from audit_source_head_validation import check_source_result


class SourceHeadValidationTests(unittest.TestCase):
    def fixture(self):
        labels = np.arange(58) % 4
        empty = np.array([], dtype=np.int64)
        pred = {
            "y_true": empty, "y_pred": empty,
            "fit_indices": np.arange(230), "validation_indices": np.arange(230, 288),
            "validation_y_true": labels, "validation_y_pred": labels,
        }
        result = {
            "seed": 1, "head_elu": False, "git_dirty": False,
            "checkpoint_selection": "final_epoch", "test_evaluation": "none",
            "test_evaluations": 0, "final_test_acc": None,
            "training_protocol": {"fit_samples": 230, "validation_samples": 58},
            "validation_fraction": 0.2, "epochs": 1000, "final_val_acc": 1.0,
            "history": [{"epoch": i, "test_acc": None, "train_loss": 0.1,
                         "train_acc": 1.0, "val_loss": 0.1, "val_acc": 1.0}
                        for i in range(1, 1001)],
        }
        return result, pred

    def test_registered_source_run_passes(self):
        result, pred = self.fixture()
        self.assertEqual(check_source_result(result, pred, False, 1), [])

    def test_any_target_evaluation_is_rejected(self):
        result, pred = self.fixture()
        result["history"][4]["test_acc"] = 0.25
        errors = check_source_result(result, pred, False, 1)
        self.assertIn("target evaluated in history", errors)

    def test_duplicate_source_index_is_rejected(self):
        result, pred = self.fixture()
        pred["validation_indices"][-1] = 0
        errors = check_source_result(result, pred, False, 1)
        self.assertIn("source indices do not partition 288 trials", errors)

    def test_validation_best_cannot_replace_final_epoch(self):
        result, pred = self.fixture()
        result["checkpoint_selection"] = "best_internal_val_accuracy"
        errors = check_source_result(result, pred, False, 1)
        self.assertTrue(any("non-final checkpoint" in e for e in errors))

    def test_non_finite_validation_loss_is_rejected(self):
        result, pred = self.fixture()
        result = copy.deepcopy(result)
        result["history"][-1]["val_loss"] = float("nan")
        errors = check_source_result(result, pred, False, 1)
        self.assertIn("missing/non-finite history val_loss", errors)

    def test_accuracy_outside_unit_interval_is_rejected(self):
        result, pred = self.fixture()
        result["history"][-1]["train_acc"] = 1.2
        errors = check_source_result(result, pred, False, 1)
        self.assertIn("history train_acc outside [0,1]", errors)


if __name__ == "__main__":
    unittest.main()
