# Week 1 Smoke Test

## Goal

Verify that the VideoMAE fine-tuning pipeline can run end-to-end without errors.

## Setup

- Dataset: UCF101
- Model: VideoMAE ViT-B
- Frames: 16
- Sampling rate: 4
- Batch size: 4
- Epochs: 1
- Evaluation: 2 clips × 3 crops

## Files

- `finetune.sh`: launch script used for the smoke test
- `train_log.txt`: raw training / evaluation log

## Outcome

- Training started successfully
- Evaluation completed successfully
- End-to-end pipeline is runnable

## Notes

This experiment is a smoke test, not the final baseline.
The next step is to run a more stable and reproducible baseline experiment.
