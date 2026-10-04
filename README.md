# AMEEGNet ours

独立实现 AMEEGNet 及跨 Session 训练实验。`--strict` 使用论文文字协议；默认配置使用训练集拟合的 Session 标准化、分支池化、轻度 dropout、AdamW、标签平滑、梯度裁剪和训练集内部验证集。

当前训练脚本每个 epoch 都计算测试准确率。strict 模式报告最终 epoch；非 strict 模式按训练 Session 内部验证集选择权重。测试曲线不能用于选择结构或参数。

## Versioning

This repository uses semantic version tags for reproducibility. Each code or
experiment-protocol change must be committed and tagged before the next run.
Generated datasets, checkpoints, and per-epoch JSON files remain local and are
excluded from Git; their summaries and audit notes are tracked.

Current version: see `VERSION`. The `--paper-pooling` flag names an exploratory
pooling/dropout hypothesis, not a verified setting from the original paper.
