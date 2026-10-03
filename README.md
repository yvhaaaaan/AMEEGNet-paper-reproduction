# AMEEGNet ours

独立实现 AMEEGNet 及跨 Session 训练实验。`--strict` 使用论文文字协议；默认配置使用训练集拟合的 Session 标准化、分支池化、轻度 dropout、AdamW、标签平滑、梯度裁剪和训练集内部验证集。

测试集只在训练结束后报告，验证集来自训练 Session。

## Versioning

This repository uses semantic version tags for reproducibility. Each code or
experiment-protocol change must be committed and tagged before the next run.
Generated datasets, checkpoints, and per-epoch JSON files remain local and are
excluded from Git; their summaries and audit notes are tracked.

Current version: `v1.0.0`
