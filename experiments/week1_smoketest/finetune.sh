#!/bin/bash
set -e

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

OUTPUT_DIR='/root/autodl-tmp/cv-research-videomae/outputs/ucf101_smoketest'
DATA_PATH='/root/autodl-tmp/cv-research-videomae/datasets/ucf101/annotations'
MODEL_PATH='/root/autodl-tmp/cv-research-videomae/pretrained/videomae/checkpoint.pth'

cd /root/autodl-tmp/cv-research-videomae/repo/third_party/VideoMAE || exit 1
mkdir -p "${OUTPUT_DIR}"

torchrun --nproc_per_node=1 --master_port=12320 run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_path "${DATA_PATH}" \
    --finetune "${MODEL_PATH}" \
    --log_dir "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --data_set UCF101 \
    --nb_classes 101 \
    --batch_size 4 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 1 \
    --num_frames 16 \
    --sampling_rate 4 \
    --num_sample 1 \
    --opt adamw \
    --lr 5e-4 \
    --warmup_lr 1e-8 \
    --min_lr 1e-5 \
    --warmup_epochs 0 \
    --layer_decay 0.7 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 1 \
    --test_num_segment 2 \
    --test_num_crop 3 \
    --fc_drop_rate 0.5 \
    --drop_path 0.2 \
    --dist_eval