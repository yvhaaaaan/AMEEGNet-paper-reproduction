# v1.0.58 Softmax/Cross-Entropy audit

## 结果

论文同时描述了最终 Softmax 和交叉熵损失，但没有明确 PyTorch 实现是把 logits 还是 Softmax 概率传给交叉熵。本轮先用 source-only 验证集进行登记后的筛选，再对 Softmax→Cross-Entropy 解释做一次完整九人 T-to-E 评估。

### Source-only screening

| Subject | Logits final source val (%) | Softmax→CE final source val (%) |
| --- | ---: | ---: |
| A01 | 77.59 | 84.48 |
| A02 | 63.79 | 63.79 |
| A04 | 77.59 | 79.31 |
| **Mean** | **72.99** | **75.86** |

目标 E Session 在该筛选阶段没有被评估。由于 Softmax→CE 的源域验证均值更高，按预注册规则继续完整九人运行。

### Nine-subject final-only run

| Subject | Final E accuracy (%) |
| --- | ---: |
| A01 | 84.03 |
| A02 | 60.07 |
| A03 | 92.36 |
| A04 | 79.51 |
| A05 | 74.31 |
| A06 | 66.32 |
| A07 | 85.76 |
| A08 | 82.64 |
| A09 | 80.90 |
| **Mean ± sample SD** | **78.43% ± 10.03%** |

与论文 `81.17% ± 10.43%` 相比，均值低 **2.74 个百分点**，标准差低 0.40 个百分点。九个目标 Session 均只在 1000 epochs 结束后评估一次；审计脚本通过，`errors` 为空。

## 判断

Softmax→CE 的确改变了优化轨迹，但没有提高九人最终均值，且略低于此前 logits 版本的 `78.67% ± 9.82%`。因此它保留为独立的论文文字协议复现记录，不替换当前主候选，也不能作为达到论文结果的证据。

固定条件：完整 576 trials、无额外滤波/伪迹删除/归一化/增强；seed 42；Adam `1e-3`；batch size 64；1000 epochs；无池化、无 dropout；ECA 位于 separable convolution 后；空间/分类器/隐藏层 max-norm `1.0/0.25/0.25`。
