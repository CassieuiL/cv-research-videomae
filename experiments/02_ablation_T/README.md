# Experiment 02: Ablation on Input Frame Number T

## 目的
研究输入帧数 T（8 / 16 / 32）对 UCF101 动作识别精度、训练耗时的影响。

## 固定配置
- Dataset: UCF101
- Pretrain: VideoMAE ViT-B (K400)
- Stride: 4, Inference: 1-clip, num_sample: 1

## 结果
| T  | Top-1  | Train Time | Batch Size |
|----|--------|------------|------------|
| 8  | 64.34% | 2h 47min   | 8          |
| 16 | 83.10% | ~9h        | 8          |
| 32 | 31.11% | 23h 12min  | 2          |

## 结论
- T=16 是精度与效率的最优平衡点
- T=8 训练最快但精度下降约 19%
- T=32 因显存限制 batch_size 降至 2，梯度不稳定，结果仅供趋势参考
