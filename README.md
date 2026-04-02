# VideoMAE Fine-tuning on UCF101 — Ablation Study

> **研究问题**：基于公开 VideoMAE 预训练权重（ViT-B），在 UCF101 上系统消融帧数 T、采样步幅 stride、推理 clip 数，并对比 K400 / SSV2 预训练权重的 domain shift 影响。

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-orange)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-UCF101-green)](https://www.crcv.ucf.edu/data/UCF101.php)

---

## 主要结果

| Pretrain | T  | Stride | Inference | Top-1  |
|----------|----|--------|-----------|--------|
| K400     | 16 | 4      | 1-clip    | 83.4%  |
| K400     | 8  | 4      | 1-clip    | 64.34% |
| K400     | 16 | 4      | 1-clip    | 83.1%  |
| K400     | 32 | 4      | 1-clip    | 31.11% |
| K400     | 16 | 2      | 1-clip    | 65.72% |
| K400     | 16 | 4      | 5-clip    | 84.67% |
| SSV2     | 16 | 4      | 1-clip    | 60.3%  |
| K400     | 16 | 4      | 1-clip    | 68.1%  |

完整结果见 [`results/tables/`](results/tables/)。

---

## 环境配置

```bash
# 1. 克隆仓库
git clone https://github.com/CassieuiL/cv-research-videomae.git
cd cv-research-videomae

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载 UCF101 数据集
# 官方地址：https://www.crcv.ucf.edu/data/UCF101.php
# 解压到 datasets/ucf101/
```

---

## 数据集准备

```
datasets/
└── ucf101/
    ├── videos/          # 原始 .avi 文件（按类别子文件夹）
    └── annotations/
        ├── trainlist01.txt
        └── testlist01.txt
```

标注文件格式（每行）：
```
ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi 1
```

---

## 预训练权重

| 来源  | 文件名                                    | 放置路径                     |
|-------|-------------------------------------------|------------------------------|
| K400  | `vit_base_patch16_224_kinetics400_800e.pth` | `pretrained_weights/`        |
| SSV2  | `vit_base_patch16_224_ssv2_800e.pth`       | `pretrained_weights/`        |

官方权重下载：https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md

---

## 快速复现

### Baseline（T=16, stride=4, 1-clip）

```bash
bash repo/scripts/finetune_ucf101_baseline.sh
```

### T 消融（Week 3）

```bash
bash repo/scripts/finetune_ucf101_T8.sh    # T=8
bash repo/scripts/finetune_ucf101_T32.sh   # T=32
```

### Stride 消融（Week 4）

```bash
bash repo/scripts/finetune_ucf101_stride2.sh
```

### 推理 clip 数消融（Week 5）

```bash
# 对已有 baseline checkpoint 做 5-clip 评测
bash repo/scripts/eval_ucf101_5clip.sh
```

### Domain Shift（Week 6）

```bash
bash repo/scripts/finetune_ucf101_ssv2_pretrain.sh   # SSV2 权重
bash repo/scripts/finetune_ucf101_k400_domain.sh     # K400 权重（对照）
```

所有脚本均使用单 GPU：

```bash
torchrun --nproc_per_node=1 run_class_finetuning.py ...
```

---

## 仓库结构

```
cv-research-videomae/
├── README.md
├── requirements.txt
├── repo/
│   ├── scripts/                  # 训练 & 评测脚本
│   │   ├── finetune_ucf101_baseline.sh
│   │   ├── finetune_ucf101_T8.sh
│   │   ├── finetune_ucf101_T32.sh
│   │   ├── finetune_ucf101_stride2.sh
│   │   ├── eval_ucf101_5clip.sh
│   │   ├── finetune_ucf101_ssv2_pretrain.sh
│   │   └── summarize_results.py
│   └── experiments/
│       ├── 01_baseline/          # Week 2
│       ├── 02_ablation_T/        # Week 3
│       ├── 03_ablation_stride/   # Week 4
│       ├── 04_ablation_infer/    # Week 5
│       └── 05_domain_shift/      # Week 6
├── results/
│   ├── tables/
│   │   ├── all_results.csv       # 汇总表
│   │   ├── baseline.csv
│   │   ├── ablation_T.csv
│   │   ├── ablation_stride.csv
│   │   ├── ablation_infer.csv
│   │   └── domain_shift.csv
│   └── figures/
│       ├── baseline_curve_ucf.png
│       ├── ablation_T_accuracy.png
│       ├── ablation_stride_accuracy.png
│       ├── tradeoff_accuracy_vs_speed.png
│       └── domain_shift_comparison.png
├── datasets/
│   └── ucf101/
├── pretrained_weights/
└── outputs/                      # 训练日志 & checkpoints（.gitignore）
```

---

## 硬件环境

| 项目       | 配置                        |
|------------|-----------------------------|
| 平台       | AutoDL 云 GPU               |
| GPU        | vGPU-32GB                   |
| 训练方式   | 单卡 `torchrun --nproc_per_node=1` |
| 数据盘     | `/root/autodl-tmp/`（跨关机持久化） |

---

## 核心结论

1. **T=16 是最优平衡点**：T=8 精度大幅下降（-19%），T=32 在单卡 32GB 下 OOM 需缩减 batch
2. **stride=4 显著优于 stride=2**（+17.68%）：UCF101 宏观动作需要更宽的时序覆盖
3. **5-clip 推理带来 +1.27% 提升，代价是 ~5× 推理时间**
4. **Domain shift 影响显著**：K400→UCF101 比 SSV2→UCF101 高 7.8 个百分点，验证了域相似度的重要性

---

## 参考文献

```bibtex
@inproceedings{tong2022videomae,
  title={VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Tong, Zhan and Song, Yibing and Wang, Jue and Wang, Limin},
  booktitle={NeurIPS},
  year={2022}
}
```

---

## 分支说明

| 分支     | 用途                   |
|----------|------------------------|
| `master` | AutoDL 服务器实验分支  |
| `main`   | 本地 Windows 整理分支  |
