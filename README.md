# AMEEGNet ours

独立实现 AMEEGNet 及跨 Session 训练实验。`--strict` 使用论文文字协议；默认配置使用训练集拟合的 Session 标准化、分支池化、轻度 dropout、AdamW、标签平滑、梯度裁剪和训练集内部验证集。

训练脚本默认只在训练结束计算一次目标 Session 测试准确率；`--test-evaluation each-epoch` 仅用于明确标注的诊断实验，不能用于选择结构、参数或 epoch。`--test-evaluation none` 可在完全不查看目标 Session 的情况下训练。非 strict 模式按训练 Session 内部验证集选择权重；strict 模式报告最终 epoch。`--deterministic` 会记录 CUDA/PyTorch 确定性设置，并保存代码 commit、数据 SHA-256、运行环境和权重指纹。

原始数据审计命令：

```text
python audit_raw_mat.py --source E:\论文\data\MNE-bnci-data\database\data-sets\001-2014 --cached data --report results\raw_mat_audit.md
```

该审计独立解析官方 MAT 文件，不调用本项目的数据加载器，并逐位核对 9 名受试者的两个 Session 与缓存 NPZ。

## Versioning

This repository uses semantic version tags for reproducibility. Each code or
experiment-protocol change must be committed and tagged before the next run.
Generated datasets, checkpoints, and per-epoch JSON files remain local and are
excluded from Git; their summaries and audit notes are tracked.

Current version: see `VERSION`. The `--paper-pooling` flag names an exploratory
pooling/dropout hypothesis, not a verified setting from the original paper.
