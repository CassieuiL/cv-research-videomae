#!/bin/bash
# ============================================================
# Week 3 Ablation T=8: UCF101 Fine-tuning
# Config: T=16, stride=4, 1-clip inference, 100 epochs
# Expected result: ~85-91% Top-1 (论文报告 91.3%)
# Usage: bash finetune_ucf101_ablation_T8.sh
# ============================================================
set -e

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

# -------- 路径配置（与你的 smoketest 保持一致）--------
OUTPUT_DIR='/root/autodl-tmp/cv-research-videomae/outputs/ucf101_ablation_T8'
DATA_PATH='/root/autodl-tmp/cv-research-videomae/datasets/ucf101/annotations'
MODEL_PATH='/root/autodl-tmp/cv-research-videomae/pretrained/videomae/checkpoint.pth'

cd /root/autodl-tmp/cv-research-videomae/repo/third_party/VideoMAE || exit 1
mkdir -p "${OUTPUT_DIR}"

# -------- 记录开始时间（用于统计训练耗时）--------
START_TIME=$(date +%s)
echo "Training started at: $(date)" | tee "${OUTPUT_DIR}/train_time.log"

torchrun --nproc_per_node=1 --master_port=12321 run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_path "${DATA_PATH}" \
    --finetune "${MODEL_PATH}" \
    --log_dir "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --data_set UCF101 \
    --nb_classes 101 \
    --batch_size 8 \
    --input_size 224 \
    --short_side_size 224 \
    --num_frames 8 \
    --sampling_rate 4 \
    --num_sample 1 \
    --opt adamw \
    --lr 5e-4 \
    --warmup_lr 1e-8 \
    --min_lr 1e-5 \
    --warmup_epochs 5 \
    --layer_decay 0.7 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 50 \
    --save_ckpt_freq 10 \
    --drop_path 0.2 \
    --fc_drop_rate 0.5 \
    --reprob 0.25 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --aa rand-m7-n4-mstd0.5-inc1 \
    --smoothing 0.1 \
    --test_num_segment 1 \
    --test_num_crop 3 \
    --dist_eval \
    2>&1 | tee "${OUTPUT_DIR}/run.log"

# -------- 记录结束时间 --------
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))

echo "" | tee -a "${OUTPUT_DIR}/train_time.log"
echo "Training finished at: $(date)" | tee -a "${OUTPUT_DIR}/train_time.log"
echo "Total training time: ${HOURS}h ${MINUTES}min" | tee -a "${OUTPUT_DIR}/train_time.log"

# -------- 从日志里提取最终 Top-1 --------
echo "" | tee -a "${OUTPUT_DIR}/train_time.log"
echo "=== Final Result ===" | tee -a "${OUTPUT_DIR}/train_time.log"
grep "Accuracy of the network" "${OUTPUT_DIR}/run.log" | tail -1 | tee -a "${OUTPUT_DIR}/train_time.log"