import unittest

import torch
from torch import nn

from audit_frozen_tf32 import compare_logits, precision_pair, verify_recorded_result
from run_s01 import state_fingerprint


class FrozenTF32Tests(unittest.TestCase):
    def test_pt_metadata_matches_its_serialized_json_types(self):
        from torch.torch_version import TorchVersion

        saved = {"betas": (0.9, 0.999), "torch": TorchVersion("1.12.1+cu116")}
        recorded = {"betas": [0.9, 0.999], "torch": "1.12.1+cu116"}
        verify_recorded_result(saved, recorded)
        with self.assertRaises(AssertionError):
            verify_recorded_result(saved, {**recorded, "betas": [0.8, 0.999]})

    def test_identical_logits_have_no_change(self):
        logits = torch.eye(4)
        result = compare_logits(logits, logits.clone(), torch.arange(4))
        self.assertTrue(result["logits_exactly_equal"])
        self.assertEqual(result["changed_prediction_trials"], 0)
        self.assertEqual(result["max_abs_logit_difference"], 0)

    def test_numerical_change_is_separate_from_prediction_change(self):
        logits = torch.eye(4)
        changed = logits.clone()
        changed[0, 0] += 0.1
        result = compare_logits(logits, changed, torch.arange(4))
        self.assertFalse(result["logits_exactly_equal"])
        self.assertEqual(result["changed_prediction_trials"], 0)
        changed[0, 1] = 2
        result = compare_logits(logits, changed, torch.arange(4))
        self.assertEqual(result["changed_prediction_trials"], 1)

    def test_non_finite_logits_are_rejected(self):
        logits = torch.eye(4)
        changed = logits.clone()
        changed[0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            compare_logits(logits, changed, torch.arange(4))

    def test_invalid_shape_and_labels_are_rejected(self):
        logits = torch.eye(4)
        for labels in (torch.arange(4).float(), torch.tensor([0, 1, 2, 4]), torch.arange(3)):
            with self.assertRaises(ValueError):
                compare_logits(logits, logits, labels)
        with self.assertRaises(ValueError):
            compare_logits(logits, logits[:3], torch.arange(4))

    def test_eval_state_and_backend_permission_are_preserved(self):
        model = nn.Sequential(nn.Linear(2, 4), nn.BatchNorm1d(4)).eval()
        fingerprint = state_fingerprint(model)
        flag = torch.backends.cudnn.allow_tf32
        a, b = precision_pair(model, torch.ones(3, 2))
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(fingerprint, state_fingerprint(model))
        self.assertEqual(flag, torch.backends.cudnn.allow_tf32)
        with self.assertRaises(ValueError):
            precision_pair(model.train(), torch.ones(3, 2))

    def test_backend_permission_restored_after_error(self):
        class Failure(nn.Module):
            def forward(self, x):
                raise RuntimeError("deliberate inference failure")

        flag = torch.backends.cudnn.allow_tf32
        with self.assertRaises(RuntimeError):
            precision_pair(Failure().eval(), torch.ones(3, 2))
        self.assertEqual(flag, torch.backends.cudnn.allow_tf32)


if __name__ == "__main__":
    unittest.main()
