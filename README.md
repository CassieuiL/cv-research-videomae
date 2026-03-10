# cv-research-videomae

This is my VideoMAE reproduction project.

The goal of this project is to build a reproducible VideoMAE fine-tuning baseline on small-scale datasets such as UCF101, and then conduct systematic ablation studies on factors including frame number, temporal stride, and inference settings, followed by lightweight domain shift and masking-related exploratory experiments.

## Current goal

Complete a minimal smoke test:

- use official pretrained weights
- choose HMDB51 or UCF101
- run training, evaluation, and curve logging

## Folder structure

- `third_party/`: official code
- `experiments/`: experiment records
- `notes/`: notes
- `scripts/`: scripts
- `data/`: dataset notes
- `pretrained/`: pretrained weight notes

### Week 1 Progress

The minimum fine-tuning pipeline has been successfully verified on UCF101 using publicly available VideoMAE pretrained weights.

Completed items:

- Set up the remote training environment on AutoDL
- Organized the UCF101 dataset and annotation files
- Loaded public VideoMAE pretrained weights
- Ran a single-GPU fine-tuning smoke test successfully
- Completed evaluation and result merging
- Obtained the first runnable baseline result

### Notes

This result is only a smoke test for verifying the full pipeline.
It is not intended to represent the final performance of the model.

### Next Steps

- Run a more stable 3-epoch baseline
- Record training time, evaluation accuracy, and configuration details
- Perform controlled ablations on:
  - number of frames
  - temporal sampling stride
  - inference clip / crop settings
- Extend to domain shift and lightweight masking trend analysis
