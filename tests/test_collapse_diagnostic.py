import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from ameegnet_ours import AMEEGNet
from audit_source_bn import construct
from audit_training_collapse import activation_stats, audit_checkpoint, inspect_source
from run_s01 import file_sha256, state_fingerprint


class CollapseDiagnosticTests(unittest.TestCase):
    def test_saturation_stat_is_not_classification_accuracy(self):
        stats = activation_stats(torch.tensor([[-25.0, 1.0], [-25.0, 3.0]]))
        self.assertEqual(stats["units_with_std_below_1e_5"], 1)
        self.assertEqual(stats["fraction_below_minus_20"], 0.5)
        self.assertTrue(stats["finite"])

    def test_source_only_inspection_preserves_state_and_removes_hooks(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = AMEEGNet(pool=True, dropout=0.25)
        inputs = np.random.default_rng(42).normal(size=(4, 22, 1125)).astype(np.float32)
        labels = np.arange(4, dtype=np.int64)
        before = state_fingerprint(model)
        report = inspect_source(model, inputs, labels, batch_size=2)
        self.assertEqual(state_fingerprint(model), before)
        self.assertEqual(report["source_true_counts"], [1, 1, 1, 1])
        self.assertEqual(sum(report["source_prediction_counts"]), 4)
        self.assertEqual(report["hidden_pre_elu"]["shape"], [4, 32])
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))
        self.assertFalse(model.training)

    def test_checkpoint_constructor_forwards_bias_and_raw_transfer(self):
        reference = AMEEGNet(fusion_pre_activation=True, conv_bias=True)
        config = {
            "strict": False, "paper_pooling": False, "dropout": 0.25,
            "fusion": True, "eca": True, "bn_first": True,
            "norm_then_activation": True, "dropout_after_pool": False,
            "head_elu": True, "head_dropout": 0.25,
            "bn_eps": 1e-5, "bn_momentum": 0.1,
            "fusion_pre_activation": True, "conv_bias": True,
        }
        restored = construct(config)
        restored.load_state_dict(reference.state_dict(), strict=True)
        self.assertTrue(restored.fusion_pre_activation)
        self.assertTrue(all(branch.temporal.bias is not None
                            for branch in (restored.b1, restored.b2, restored.b3)))

    def test_mutated_buffer_is_rejected(self):
        torch.set_num_threads(1)
        model = AMEEGNet(pool=True)

        def mutate(module, args, output):
            model.b1.bn_d.running_mean.add_(1)

        handle = model.register_forward_hook(mutate)
        try:
            with self.assertRaisesRegex(AssertionError, "changed model"):
                inspect_source(model, np.zeros((2, 22, 1125), dtype=np.float32),
                               np.array([0, 1], dtype=np.int64))
        finally:
            handle.remove()

    def test_mismatched_data_or_source_hash_is_rejected(self):
        repository = Path(__file__).resolve().parents[1]
        hashes = {name: file_sha256(repository / name) for name in (
            "run_s01.py", "ameegnet_ours/model.py", "ameegnet_ours/data.py",
        )}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.npz"
            data.write_bytes(b"not loaded because the hash guard rejects it")
            checkpoint = root / "checkpoint.pt"
            config = {
                "strict": True, "reverse_sessions": False,
                "validation_fraction": None, "data_path": str(data),
                "data_sha256": "incorrect", "source_sha256": hashes,
            }
            torch.save({"result": config}, checkpoint)
            with self.assertRaisesRegex(AssertionError, "data hash differs"):
                audit_checkpoint(checkpoint, 32)
            config["source_sha256"] = dict(hashes, **{"run_s01.py": "incorrect"})
            torch.save({"result": config}, checkpoint)
            with self.assertRaisesRegex(AssertionError, "training source hash differs"):
                audit_checkpoint(checkpoint, 32)


if __name__ == "__main__":
    unittest.main()
