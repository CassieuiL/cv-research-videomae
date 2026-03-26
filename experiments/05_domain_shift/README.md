# Experiment 05: Domain Shift Analysis

## 研究问题
不同预训练来源（K400 vs SSV2）迁移到 UCF101 的效果差异？

## 实验配置
- Fine-tune 配置完全一致：T=16, stride=4, 50 epochs, batch_size=8
- 只改变预训练权重来源

## 结果
| 预训练 | Top-1 |
|--------|-------|
| K400   | 68.1% |
| SSV2   | 60.3% |

## 结论
K400→UCF101 比 SSV2→UCF101 高 7.8%。
K400 与 UCF101 同属"场景相关动作"数据集，domain 接近。
SSV2 以手物交互为主，与 UCF101 存在明显 domain gap。
