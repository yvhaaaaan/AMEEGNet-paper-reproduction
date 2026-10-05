# AMEEGNet v1.0.47 九人消融审计

日期：2026-10-06  
代码提交：`4fe043c966bb01d8cf1c81bc78870feece85410c`  
运行环境：Python 3.10.21，PyTorch 1.12.1+cu116，RTX 4070 Laptop GPU  
固定协议：seed=42，T session 训练、E session 测试，22 通道，`[1.5, 6] s`，四分类，Adam `lr=0.001`，batch size=64，1000 epochs，最终 epoch 只评估一次测试集。

## 九人汇总

| 结构 | 本次均值 | 本次样本 SD | 论文 Table 3 均值 | 论文样本 SD | 与论文差值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parallel EEGNet | 75.54% | 12.31% | 78.32% | 9.85% | -2.78 pp |
| Parallel + Fusion | 75.81% | 13.80% | 79.40% | 10.40% | -3.59 pp |
| Parallel + ECA | 73.73% | 13.13% | 79.01% | 11.22% | -5.28 pp |
| AMEEGNet（Fusion + ECA） | 76.35% | 11.36% | 81.17% | 10.43% | -4.82 pp |

论文报告的 Table 3 数值来自多受试者 BCI IV 2a 消融实验；本表使用同一数据集、同一九名受试者和同一最终 epoch 口径，但仍是本地重建实现的复现审计，不是作者代码的逐行复现。

## 逐受试者结果

| Subject | Parallel | + Fusion | + ECA | Full AMEEGNet |
| --- | ---: | ---: | ---: | ---: |
| A01 | 85.76% | 86.46% | 80.21% | 85.42% |
| A02 | 57.99% | 51.74% | 54.51% | 60.42% |
| A03 | 93.06% | 90.28% | 91.32% | 92.36% |
| A04 | 70.14% | 72.22% | 63.89% | 71.18% |
| A05 | 74.31% | 69.10% | 72.22% | 72.92% |
| A06 | 56.94% | 60.07% | 55.90% | 59.72% |
| A07 | 85.42% | 88.54% | 86.11% | 86.46% |
| A08 | 79.86% | 88.19% | 82.64% | 80.90% |
| A09 | 76.39% | 75.69% | 76.74% | 78.82% |

## 解释与限制

- 当前实现的相对趋势为：Fusion 单独略高于 Parallel，ECA 单独低于 Parallel，Fusion+ECA 高于三个当前消融均值但仍低于论文 Full AMEEGNet。这与论文 Table 3 的“加入 ECA 后提升”不一致，因此不能宣称 AMEEGNet 原文复现已经通过。
- 所有结果均为测试集最终一次评估；没有用测试集选择 epoch、结构或超参数。
- 本批次的训练代码和模型代码哈希与 v1.0.46 一致；本版本只归档结果，不修改算法。
- 结果目录：
  - `results/all_ablation_parallel_no_fusion_no_eca_v1_0_46`
  - `results/all_ablation_parallel_fusion_no_eca_v1_0_46`
  - `results/all_ablation_parallel_eca_no_fusion_v1_0_46`
  - `results/all_dropout025_hiddenmax_pytorch112_v1_0_38`

## 下一步

优先做不改变网络结构的单因素审计：数据 Session 方向、论文图中池化/Dropout 的确切位置，以及训练过程中是否存在作者未写出的 checkpoint 或随机性处理。任何改动都需要先在 A01 记录，再扩展到九人，并继续保留最终 epoch 与训练过程最佳值的区分。
