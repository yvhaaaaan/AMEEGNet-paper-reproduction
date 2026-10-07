# AMEEGNet ours

独立实现 AMEEGNet 及跨 Session 训练实验。`--strict` 固定论文明确给出的部分训练设置，但不代表未披露的结构和参数已得到原作者确认；默认配置使用训练集拟合的 Session 标准化、分支池化、轻度 dropout、AdamW、标签平滑、梯度裁剪和训练集内部验证集。

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

## Reproduction Status

The retained exploratory nine-subject final-epoch candidate is
`78.6651% +/- 9.8180%` (sample SD), below the paper's `81.17%` mean by
`2.5049` percentage points. This is not exact recovery of the author's code:
several implementation choices are unspecified, max-norm is an assumption,
and the broader reconstruction history includes repeated target-labelled
diagnostics. Do not select a test-best epoch or omit low seed results.

See [the current audit](results/v1_0_83_reproduction_status_20261007.md) and
[the seed sensitivity report](results/v1_0_75_seed_sensitivity_audit_20261007.md).
The [paired source stability check](results/v1_0_78_source_head_stability_audit_20261007.md)
isolates the unspecified hidden ELU in two collapse conditions. Its 100%
source-fit results are not held-out decoding results; the primary observed
target result is unchanged.

The [paired source-validation gate](results/v1_0_80_source_head_validation_audit_20261007.md)
completed: hidden ELU77.4425%, noELU77.0115%, paired difference-0.4310pp.
Integrity QA passed, but the performance gate failed; the noELU target batch
was not launched. The training-mode stability finding is not a validation gain.

The [frozen cuDNN precision diagnosis](results/v1_0_82_frozen_tf32_audit_20261007.md)
reproduced all twelve controls under both permissions:0/696 source predictions
changed. This is not a training-accuracy result and does not override the gate.
