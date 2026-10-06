import math

import torch
from torch import nn
import torch.nn.functional as F


def same_time(x, kernel):
    left = kernel // 2 - 1 if kernel % 2 == 0 else kernel // 2
    right = kernel // 2
    return F.pad(x, (left, right, 0, 0))


class ECA(nn.Module):
    def __init__(self, channels, kernel=3, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel, padding=kernel // 2, bias=bias)

    def forward(self, x):
        w = x.mean(dim=(2, 3), keepdim=False).unsqueeze(1)
        w = torch.sigmoid(self.conv(w)).transpose(1, 2).unsqueeze(-1)
        return x * w


class Branch(nn.Module):
    def __init__(self, channels, f1, kernel, depth_in=None, sep_in=None,
                 pool=True, dropout=0.25, bn_first=True, norm_then_activation=True,
                 dropout_after_pool=False, bn_eps=1e-5, bn_momentum=0.1,
                 depth_out=None, depth_groups=None, fusion_pre_activation=False,
                 conv_bias=False):
        super().__init__()
        depth_in = depth_in or f1
        sep_in = sep_in or depth_in * 2
        self.temporal = nn.Conv2d(1, f1, (1, kernel), bias=conv_bias)
        self.bn_t = nn.BatchNorm2d(f1, eps=bn_eps, momentum=bn_momentum) if bn_first else nn.Identity()
        depth_out = depth_in * 2 if depth_out is None else depth_out
        depth_groups = depth_in if depth_groups is None else depth_groups
        if depth_out % depth_groups != 0:
            raise ValueError("depth_out must be divisible by depth_groups")
        self.depth = nn.Conv2d(depth_in, depth_out, (channels, 1),
                               groups=depth_groups, bias=conv_bias)
        self.bn_d = nn.BatchNorm2d(depth_out, eps=bn_eps, momentum=bn_momentum)
        self.sep_dw = nn.Conv2d(sep_in, sep_in, (1, 16), groups=sep_in, bias=conv_bias)
        self.sep_pw = nn.Conv2d(sep_in, f1 * 2, 1, bias=conv_bias)
        self.bn_s = nn.BatchNorm2d(f1 * 2, eps=bn_eps, momentum=bn_momentum)
        self.norm_then_activation = norm_then_activation
        self.fusion_pre_activation = fusion_pre_activation
        self.dropout_after_pool = dropout_after_pool
        self.pool1 = nn.AvgPool2d((1, 4)) if pool else nn.Identity()
        self.pool2 = nn.AvgPool2d((1, 8)) if pool else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

    def temporal_out(self, x):
        return self.bn_t(self.temporal(same_time(x, self.temporal.kernel_size[1])))

    def from_temporal(self, h, temporal_fusion=None, depth_fusion=None,
                      depth_gate=None, depth_pre_activation=False,
                      sep_gate=None):
        if temporal_fusion is not None:
            h = torch.cat((temporal_fusion, h), dim=1)
        h = self.depth(h)
        depth_raw = h
        h = self.bn_d(h) if self.norm_then_activation else self.bn_d(F.elu(h))
        h = F.elu(h) if self.norm_then_activation else h
        if not self.dropout_after_pool:
            h = self.drop(h)
        depth_out = depth_raw if depth_pre_activation else h
        if depth_gate is not None:
            h = depth_gate(h)
            # Keep the requested pre-activation tensor for the next fusion
            # junction; ECA gates the local branch output that continues to
            # the separable convolution.
            if not depth_pre_activation:
                depth_out = h
        if depth_fusion is not None:
            h = torch.cat((depth_fusion, h), dim=1)
        h = self.pool1(h)
        if self.dropout_after_pool:
            h = self.drop(h)
        h = same_time(h, 16)
        h = self.sep_pw(self.sep_dw(h))
        h = self.bn_s(h) if self.norm_then_activation else self.bn_s(F.elu(h))
        h = F.elu(h) if self.norm_then_activation else h
        if sep_gate is not None:
            h = sep_gate(h)
        if not self.dropout_after_pool:
            h = self.drop(h)
        h = self.pool2(h)
        if self.dropout_after_pool:
            h = self.drop(h)
        return h, depth_out

    def forward(self, x, temporal_fusion=None, depth_fusion=None,
                depth_gate=None):
        return self.from_temporal(self.temporal_out(x), temporal_fusion,
                                  depth_fusion, depth_gate)


class AMEEGNet(nn.Module):
    def __init__(self, channels=22, samples=1125, classes=4,
                 pool=True, dropout=0.25, fusion=True, eca=True,
                 bn_first=True, norm_then_activation=True,
                 dropout_after_pool=False, head_elu=True,
                 bn_eps=1e-5, bn_momentum=0.1, head_dropout=None,
                 eca_bias=False, init_mode="default", eca_stage="output",
                 fixed_fusion_channels=False, fusion_pre_activation=False,
                 conv_bias=False):
        super().__init__()
        if not math.isfinite(bn_eps) or bn_eps <= 0:
            raise ValueError("bn_eps must be finite and positive")
        if not math.isfinite(bn_momentum) or not 0 < bn_momentum <= 1:
            raise ValueError("bn_momentum must be finite and in (0,1]")
        if head_dropout is None:
            head_dropout = dropout
        if not math.isfinite(head_dropout) or not 0 <= head_dropout < 1:
            raise ValueError("head_dropout must be finite and in [0,1)")
        if init_mode not in ("default", "xavier_uniform", "xavier_normal",
                             "kaiming_normal"):
            raise ValueError(f"unsupported init_mode: {init_mode}")
        if eca_stage not in ("output", "depth_pre_sep", "sep_pre_pool"):
            raise ValueError(f"unsupported eca_stage: {eca_stage}")
        self.fusion = fusion
        self.use_eca = eca
        self.eca_stage = eca_stage
        self.fixed_fusion_channels = fixed_fusion_channels
        self.fusion_pre_activation = fusion_pre_activation
        self.b1 = Branch(channels, 4, 16, pool=pool, dropout=dropout,
                         bn_first=bn_first, norm_then_activation=norm_then_activation,
                         dropout_after_pool=dropout_after_pool,
                         bn_eps=bn_eps, bn_momentum=bn_momentum,
                         fusion_pre_activation=fusion_pre_activation,
                         conv_bias=conv_bias)
        b2_depth_in = 12 if fusion else 8
        if fusion and fixed_fusion_channels:
            b2_depth_out = 16
            b2_depth_groups = 1
            b2_sep_in = 16
            b3_sep_in = 48
        else:
            b2_depth_out = None
            b2_depth_groups = None
            b2_sep_in = 24 if fusion else 16
            b3_sep_in = 56 if fusion else 32
        self.b2 = Branch(channels, 8, 32, depth_in=b2_depth_in, sep_in=b2_sep_in,
                         depth_out=b2_depth_out, depth_groups=b2_depth_groups,
                         pool=pool, dropout=dropout, bn_first=bn_first,
                         norm_then_activation=norm_then_activation,
                         dropout_after_pool=dropout_after_pool,
                         bn_eps=bn_eps, bn_momentum=bn_momentum,
                         fusion_pre_activation=fusion_pre_activation,
                         conv_bias=conv_bias)
        self.b3 = Branch(channels, 16, 64, depth_in=16, sep_in=b3_sep_in,
                         pool=pool, dropout=dropout, bn_first=bn_first,
                         norm_then_activation=norm_then_activation,
                         dropout_after_pool=dropout_after_pool,
                         bn_eps=bn_eps, bn_momentum=bn_momentum,
                         fusion_pre_activation=fusion_pre_activation,
                         conv_bias=conv_bias)
        if eca_stage in ("output", "sep_pre_pool"):
            eca_channels = [8, 16, 32]
        else:
            eca_channels = [self.b1.depth.out_channels,
                            self.b2.depth.out_channels,
                            self.b3.depth.out_channels]
        self.attn = nn.ModuleList([ECA(c, bias=eca_bias) for c in eca_channels])
        # All branches preserve time in their convolutions.  Pooling therefore
        # changes 1125 to floor((floor((1125-4)/4+1)-8)/8+1)=35.
        # Infer the head width analytically so construction does not update BN
        # running statistics or consume Dropout RNG state with a dummy batch.
        if pool:
            pooled = (samples - 4) // 4 + 1
            pooled = (pooled - 8) // 8 + 1
        else:
            pooled = samples
        n = (8 + 16 + 32) * pooled
        head = [nn.Linear(n, 32)]
        if head_elu:
            head.append(nn.ELU())
        head.extend([nn.Dropout(head_dropout), nn.Linear(32, classes)])
        self.head = nn.Sequential(*head)
        self.init_mode = init_mode
        if init_mode != "default":
            self._initialize_weights(init_mode)

    def _initialize_weights(self, mode):
        """Apply an explicit initialization for an architecture audit."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                if mode == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif mode == "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                elif mode == "kaiming_normal":
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _features(self, x):
        t1 = self.b1.temporal_out(x)
        t2 = self.b2.temporal_out(x)
        t3 = self.b3.temporal_out(x)
        if self.use_eca and self.eca_stage == "depth_pre_sep":
            h1, _ = self.b1.from_temporal(
                t1, depth_gate=self.attn[0],
                depth_pre_activation=self.fusion_pre_activation)
            h2, d2 = self.b2.from_temporal(
                t2, temporal_fusion=t1 if self.fusion else None,
                depth_gate=self.attn[1],
                depth_pre_activation=self.fusion_pre_activation)
            h3, _ = self.b3.from_temporal(
                t3, depth_fusion=d2 if self.fusion else None,
                depth_gate=self.attn[2],
                depth_pre_activation=self.fusion_pre_activation)
            return [h1, h2, h3]
        if self.use_eca and self.eca_stage == "sep_pre_pool":
            h1, _ = self.b1.from_temporal(
                t1, sep_gate=self.attn[0],
                depth_pre_activation=self.fusion_pre_activation)
            h2, d2 = self.b2.from_temporal(
                t2, temporal_fusion=t1 if self.fusion else None,
                sep_gate=self.attn[1],
                depth_pre_activation=self.fusion_pre_activation)
            h3, _ = self.b3.from_temporal(
                t3, depth_fusion=d2 if self.fusion else None,
                sep_gate=self.attn[2],
                depth_pre_activation=self.fusion_pre_activation)
            return [h1, h2, h3]
        h1, _ = self.b1.from_temporal(
            t1, depth_pre_activation=self.fusion_pre_activation)
        h2, d2 = self.b2.from_temporal(
            t2, temporal_fusion=t1 if self.fusion else None,
            depth_pre_activation=self.fusion_pre_activation)
        h3, _ = self.b3.from_temporal(
            t3, depth_fusion=d2 if self.fusion else None,
            depth_pre_activation=self.fusion_pre_activation)
        out = [h1, h2, h3]
        return [a(o) for a, o in zip(self.attn, out)] if self.use_eca else out

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.shape[1:] != (1, 22, 1125):
            raise ValueError(f"expected (N,22,1125), got {tuple(x.shape)}")
        return self.head(torch.cat(self._features(x), dim=1).flatten(1))

    @torch.no_grad()
    def project_eegnet_max_norm(self, spatial_max=1.0, classifier_max=0.25,
                                hidden_max=None):
        """Apply the standard EEGNet kernel constraints after an optimizer step."""
        for branch in (self.b1, self.b2, self.b3):
            weight = branch.depth.weight
            norms = weight.flatten(1).norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            weight.mul_((spatial_max / norms).clamp(max=1.0).view(-1, 1, 1, 1))
        weight = self.head[-1].weight
        norms = weight.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        weight.mul_((classifier_max / norms).clamp(max=1.0))
        if hidden_max is not None:
            weight = self.head[0].weight
            norms = weight.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            weight.mul_((hidden_max / norms).clamp(max=1.0))
