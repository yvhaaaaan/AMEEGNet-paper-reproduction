# v1.0.57 官方伪迹标记剔除敏感性分析

## 结论

本轮使用 BCI Competition IV 2a 原始 MAT 文件中的官方伪迹标记，删除标记为非零的 trial 后重新训练。该协议与 AMEEGNet 论文“仅分段”的主协议不同，因此结果只用于数据协议敏感性分析，不作为原论文复现结果。

9 名受试者均完成 1000 epochs 训练，最终测试一次，使用第 1000 epoch 模型；没有使用目标测试集选 epoch。JSON、NPZ 和 checkpoint 文件均存在，预测标签与真实标签重新计算的准确率一致，且每名受试者的四类标签均存在。

| Subject | Train trials | Test trials | Final accuracy (%) | Time (s) |
| --- | ---: | ---: | ---: | ---: |
| A01 | 273 | 281 | 84.70 | 197.43 |
| A02 | 270 | 283 | 67.14 | 196.88 |
| A03 | 270 | 273 | 87.18 | 196.04 |
| A04 | 262 | 228 | 81.58 | 196.31 |
| A05 | 262 | 276 | 67.39 | 189.48 |
| A06 | 219 | 215 | 60.00 | 154.09 |
| A07 | 271 | 277 | 85.56 | 192.68 |
| A08 | 264 | 271 | 72.32 | 188.67 |
| A09 | 237 | 264 | 81.44 | 158.96 |

**Mean ± sample SD:** **76.37% ± 9.83%**  
**Difference from paper mean 81.17%:** **−4.80 percentage points**  
Runtime mean: 185.62 s; median: 192.68 s.

## 固定配置

- Git result code: `dff4de2703b031431bbc6af19bd8758cd68502ed` (`v1.0.56`)
- Audit code: `c6323155445b7ea0a3199229bedf56d0a40e91c2` (`v1.0.57`)
- Seed: 42; deterministic CUDA; RTX 4070 Laptop GPU
- Adam, learning rate `0.001`, batch size `64`, `1000` epochs
- Final epoch evaluation only; no validation split; no test-based model selection
- No v1.7 enhancement, no extra filtering, no normalization, no augmentation
- `input_scale=1.0`, no pooling, no dropout, ECA at `sep_pre_pool`
- Weight constraints: spatial `1.0`, classifier `0.25`, hidden `0.25`

## 审计说明

由于删除伪迹 trial 后各受试者有效 trial 数不同，审计脚本比较配置时排除了 subject-specific 的 trial count，并逐人核对实际索引数量。测试集不再强制要求每类 72 个样本，但强制检查四类均存在。`audit.stdout.json` 中 `errors` 为空。

这轮结果不能用于声称“AMEEGNet 已达到论文结果”。主复现仍应使用保留全部 trial 的完整 576-trial 数据协议；当前最好完整协议批处理结果仍为 78.67% ± 9.82%，低于论文 81.17% ± 10.43%。
