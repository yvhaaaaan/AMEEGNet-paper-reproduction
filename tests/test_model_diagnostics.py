import math
import unittest

import torch
import numpy as np
from torch import nn

from ameegnet_ours import AMEEGNet
from ameegnet_ours.model import Branch
from run_s01 import normalize_trials
from audit_source_bn import recalibrate


class ModelDiagnosticsTests(unittest.TestCase):
    def test_default_audit_switches_preserve_initialization(self):
        torch.manual_seed(42)
        implicit = AMEEGNet()
        torch.manual_seed(42)
        explicit = AMEEGNet(init_mode="default", eca_stage="output",
                            fixed_fusion_channels=False, conv_bias=False)
        for name, tensor in implicit.state_dict().items():
            self.assertTrue(torch.equal(tensor, explicit.state_dict()[name]), name)

    def test_architecture_audit_shapes_and_gradients(self):
        torch.set_num_threads(1)
        settings = ({"eca_stage": "depth_pre_sep"},
                    {"eca_stage": "depth_pre_sep", "fusion_pre_activation": True},
                    {"eca_stage": "sep_pre_pool"},
                    {"conv_bias": True},
                    {"fixed_fusion_channels": True},
                    {"init_mode": "xavier_uniform"},
                    {"init_mode": "xavier_normal"},
                    {"init_mode": "kaiming_normal"})
        for config in settings:
            with self.subTest(config=config):
                model = AMEEGNet(**config)
                logits = model(torch.randn(2, 22, 1125))
                self.assertEqual(tuple(logits.shape), (2, 4))
                logits.square().mean().backward()
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item())
        for config in ({"init_mode": "unsupported"}, {"eca_stage": "unsupported"}):
            with self.assertRaises(ValueError):
                AMEEGNet(**config)

    def test_fusion_pre_activation_switch_changes_sep_path(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        post = AMEEGNet(eca_stage="sep_pre_pool", fusion_pre_activation=False)
        torch.manual_seed(42)
        pre = AMEEGNet(eca_stage="sep_pre_pool", fusion_pre_activation=True)
        for name, tensor in post.state_dict().items():
            self.assertTrue(torch.equal(tensor, pre.state_dict()[name]), name)
        inputs = torch.randn(2, 22, 1125)
        post.eval()
        pre.eval()
        with torch.inference_mode():
            post_logits = post(inputs)
            pre_logits = pre(inputs)
        self.assertFalse(torch.equal(post_logits, pre_logits))

    def test_fusion_pre_activation_switch_changes_depth_eca_path(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        post = AMEEGNet(eca_stage="depth_pre_sep", fusion_pre_activation=False)
        torch.manual_seed(42)
        pre = AMEEGNet(eca_stage="depth_pre_sep", fusion_pre_activation=True)
        for name, tensor in post.state_dict().items():
            self.assertTrue(torch.equal(tensor, pre.state_dict()[name]), name)
        inputs = torch.randn(2, 22, 1125)
        post.eval()
        pre.eval()
        with torch.inference_mode():
            post_logits = post(inputs)
            pre_logits = pre(inputs)
        self.assertFalse(torch.equal(post_logits, pre_logits))

    def test_raw_fusion_return_does_not_disable_local_eca(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        branch = Branch(22, 4, 16, dropout=0).eval()
        temporal = torch.randn(2, 4, 22, 1125)
        gate = lambda x: x * 0.5
        with torch.inference_mode():
            raw = branch.depth(temporal)
            local_pre, transfer_pre = branch.from_temporal(
                temporal, depth_gate=gate, depth_pre_activation=True)
            local_post, transfer_post = branch.from_temporal(
                temporal, depth_gate=gate, depth_pre_activation=False)
            expected_post = gate(nn.functional.elu(branch.bn_d(raw)))
        self.assertTrue(torch.equal(transfer_pre, raw))
        self.assertTrue(torch.equal(transfer_post, expected_post))
        self.assertTrue(torch.equal(local_pre, local_post))

    def test_raw_transfer_joins_gated_local_b3_features(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = AMEEGNet(pool=False, dropout=0, bn_first=False,
                         eca_stage="depth_pre_sep", fusion_pre_activation=True).eval()
        captured = {}

        def output_hook(name):
            def capture(module, inputs, output):
                captured[name] = output.detach().clone()
            return capture

        def input_hook(module, inputs):
            captured["b3_fused"] = inputs[0].detach().clone()

        handles = [
            model.b2.depth.register_forward_hook(output_hook("b2_raw")),
            model.attn[2].register_forward_hook(output_hook("b3_gated")),
            model.b3.pool1.register_forward_pre_hook(input_hook),
        ]
        try:
            with torch.inference_mode():
                model(torch.randn(2, 22, 1125))
        finally:
            for handle in handles:
                handle.remove()
        expected = torch.cat((captured["b2_raw"], captured["b3_gated"]), dim=1)
        self.assertEqual(tuple(expected.shape), (2, 56, 1, 1125))
        self.assertTrue(torch.equal(captured["b3_fused"], expected))

    def test_source_bn_diagnostic_never_changes_weights(self):
        torch.set_num_threads(1)
        for mode in ("single_source_batch", "sequential_source_moments"):
            with self.subTest(mode=mode):
                model = AMEEGNet(bn_first=False)
                before = {name: p.clone() for name, p in model.named_parameters()}
                recalibrate(model, torch.randn(4, 22, 1125), mode)
                self.assertFalse(model.training)
                for name, parameter in model.named_parameters():
                    self.assertTrue(torch.equal(parameter, before[name]), name)
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        self.assertTrue(torch.isfinite(module.running_mean).all())
                        self.assertTrue((module.running_var > 0).all())

    def test_head_dropout_default_preserves_state(self):
        torch.manual_seed(42)
        implicit = AMEEGNet(pool=True, dropout=0.5)
        torch.manual_seed(42)
        explicit = AMEEGNet(pool=True, dropout=0.5, head_dropout=0.5)
        self.assertEqual(implicit.head[-2].p, 0.5)
        for name, tensor in implicit.state_dict().items():
            self.assertTrue(torch.equal(tensor, explicit.state_dict()[name]), name)

    def test_head_dropout_switch_does_not_change_branches(self):
        model = AMEEGNet(pool=True, dropout=0.5, head_dropout=0)
        self.assertEqual(model.head[-2].p, 0)
        for branch in (model.b1, model.b2, model.b3):
            self.assertEqual(branch.drop.p, 0.5)

    def test_bn_factors_and_validation(self):
        model = AMEEGNet(bn_first=False, bn_eps=0.001, bn_momentum=0.01)
        modules = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        self.assertEqual(len(modules), 6)
        self.assertTrue(all(module.eps == 0.001 and module.momentum == 0.01 for module in modules))
        for values in ({"bn_eps": 0}, {"bn_eps": math.nan},
                       {"bn_momentum": 0}, {"bn_momentum": 1.1},
                       {"head_dropout": -0.1}, {"head_dropout": 1},
                       {"head_dropout": math.nan}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                AMEEGNet(**values)

    def test_eca_bias_switch(self):
        no_bias = AMEEGNet(eca_bias=False)
        with_bias = AMEEGNet(eca_bias=True)
        self.assertTrue(all(module.conv.bias is None for module in no_bias.attn))
        self.assertTrue(all(module.conv.bias is not None for module in with_bias.attn))

    def test_branch_conv_bias_is_independent_of_eca_bias(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                model = AMEEGNet(conv_bias=enabled, eca_bias=False)
                for module in model.modules():
                    if isinstance(module, nn.Conv2d):
                        self.assertEqual(module.bias is not None, enabled)
                self.assertTrue(all(module.conv.bias is None for module in model.attn))

    def test_trial_normalization_is_finite_and_scoped(self):
        x = np.arange(2 * 22 * 5, dtype=np.float32).reshape(2, 22, 5)
        self.assertTrue(np.array_equal(normalize_trials(x, "none"), x))
        trial = normalize_trials(x, "trial")
        channel = normalize_trials(x, "channel-trial")
        self.assertTrue(np.allclose(trial.mean(axis=(1, 2)), 0))
        self.assertTrue(np.allclose(trial.std(axis=(1, 2)), 1))
        self.assertTrue(np.allclose(channel.mean(axis=2), 0, atol=1e-6))
        self.assertTrue(np.allclose(channel.std(axis=2), 1, atol=1e-6))

    def test_finite_forward_backward_and_eval_state(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        model = AMEEGNet(pool=True, dropout=0.5, head_dropout=0,
                         bn_first=False, dropout_after_pool=True)
        inputs = torch.randn(2, 22, 1125)
        targets = torch.tensor([0, 1])
        model.train()
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (2, 4))
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        model.eval()
        with torch.inference_mode():
            first, second = model(inputs), model(inputs)
        self.assertTrue(torch.equal(first, second))
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[name]), name)

    def test_loader_rng_audit_is_exposed(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "run_s01.py", "--help"],
            check=True, capture_output=True, text=True
        )
        self.assertIn("--loader-rng", result.stdout)


if __name__ == "__main__":
    unittest.main()
