# v1.0.60 DataLoader RNG audit

## Source-only screening

普通全局 RNG batch 顺序与独立 generator 的 source-only 结果如下。目标 E
Session 在筛选阶段未被评估。

| Subject | Isolated generator (%) | Global generator (%) |
| --- | ---: | ---: |
| A01 | 77.59 | 81.03 |
| A02 | 63.79 | 65.52 |
| A04 | 77.59 | 79.31 |
| **Mean** | **72.99** | **75.86** |

global generator 的 source-only 均值更高，因此按预登记规则继续完整九人最终测试。

## Nine-subject final-only run

| Subject | Final E accuracy (%) |
| --- | ---: |
| A01 | 85.07 |
| A02 | 67.01 |
| A03 | 90.97 |
| A04 | 80.90 |
| A05 | 72.22 |
| A06 | 64.24 |
| A07 | 85.42 |
| A08 | 77.08 |
| A09 | 81.25 |
| **Mean ± sample SD** | **78.24% ± 8.92%** |

与论文 `81.17% ± 10.43%` 相比，均值低 **2.93 个百分点**；与当前 isolated
generator 主候选 `78.67% ± 9.82%` 相比低 0.43 个百分点。9 名受试者均只在
1000 epochs 结束后评估一次，审计脚本通过。

## 判断

DataLoader 的 RNG 来源会改变优化轨迹和个体结果，但没有提高九人最终均值，
因此不替换当前主候选。该批次未使用自研增强、滤波、伪迹删除、测试集选 epoch
或测试集选参数。
