# AMEEGNet v1.0.52 结构复核补充

日期：2026-10-06  
代码提交：`a3bfec7739ac4db35adbdb6b52c21d4b0bae4ab1`  
共同协议：T Session 训练、E Session 测试；1000 epochs；Adam 0.001；batch size 64；严格论文模式；训练期间不评估目标 Session；最终只测试一次；无自研增强。

## 结果

| 运行 | 关键差异 | 均值 ± 样本 SD | 与论文均值差 |
| --- | --- | ---: | ---: |
| no-pool/no-dropout, seed=2025 | ECA after separable conv | 76.23% ± 9.06% | -4.94 pp |
| no-pool/no-dropout, seed=42 | ECA after separable conv | 78.67% ± 9.82% | -2.50 pp |
| no-pool/no-dropout, seed=42 | ECA after depthwise conv | 78.09% ± 10.09% | -3.08 pp |
| no-pool/no-dropout, seed=42 | fixed fusion output widths | 77.62% ± 10.05% | -3.55 pp |

论文 BCI IV 2a 报告为 `81.17% ± 10.43%`。

## 结构判断

Figure 2 未画出池化和 Dropout；融合后的分支标注仍保持 `F1_i×D` 输出宽度。因此 no-pool/no-dropout 是图示上更接近的候选，`fixed_fusion_channels` 是通道标注的直接字面实现。两种结构复核均低于默认融合宽度的 no-pool/no-dropout 候选，未替换当前候选。

ECA 位于深度卷积之后的正文文字版本也已完成 9 人运行，但均值较低。当前较好结果仍是 seed=42 的 no-pool/no-dropout、ECA after separable conv：`78.67% ± 9.82%`；它仍不能宣称达到原论文结果。

## 审计

三组九人目录均通过 `audit_hidden_maxnorm_batch.py`：9/9 记录齐全，配置和源文件哈希一致，history 长度均为 1000，训练期间 `test_acc` 全为空，目标测试次数均为 1，JSON 与预测标签准确率一致。

