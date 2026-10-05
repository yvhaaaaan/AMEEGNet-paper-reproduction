import math
import unittest

import torch
from torch import nn

from ameegnet_ours import AMEEGNet


class ModelDiagnosticsTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
