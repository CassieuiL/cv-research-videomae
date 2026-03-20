# Experiment 03: Ablation on Temporal Stride (Sampling Rate)

## 目的
研究采样步幅 stride（2 vs 4）对 UCF101 动作识别精度与训练耗时的影响。

## 固定配置
- Dataset: UCF101
- Pretrain: VideoMAE ViT-B (K400)
- T: 16, Inference: 1-clip, Batch size: 8

## 结果
| Stride | Top-1 (final) | Top-1 (best) | Train Time | Epochs |
|--------|--------------|--------------|------------|--------|
| 4      | 83.10%       | 83.40%       | 14h 32min  | 100    |
| 2      | 65.20%       | 65.72%       | 7h 19min   | 50     |

## 结论
- stride=4 显著优于 stride=2，差距约 17.7%
- UCF101 动作以宏观动作为主，stride=4 覆盖更长时间跨度，能捕捉完整动作
- stride=2 视野过短，仅看到局部密集帧，丢失全局时序信息
- 结论与 VideoMAE 论文在 K400 上使用 stride=4 的设计选择一致
