# Experiment 01: UCF101 Baseline

## 目的
建立 UCF101 fine-tune baseline，作为后续消融实验（T/stride/infer clip）的对照组。

## 配置
- Pretrain: VideoMAE ViT-B (K400, 3200 epochs)
- T=16, stride=4, batch_size=8, num_sample=1, inference=1 clip
- Epochs: 100

## 结果
- Top-1 (epoch 100): 83.1%
- Top-1 (best checkpoint): 83.4% @ epoch 86

## 与论文差距说明
论文报告 91.3%，差距约 8%。
原因：batch_size 8（论文 128）+ 关闭 repeated augmentation，单卡资源限制下预期结果。
消融实验的相对结论不受影响。
