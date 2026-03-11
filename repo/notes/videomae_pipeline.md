# VideoMAE Pipeline Notes

## Official repo

https://github.com/MCG-NJU/VideoMAE

## Things I need to figure out

- where the fine-tuning script is
- where the evaluation script is
- how the dataset should be organized
- where to put pretrained weights
- how logging is done
- where the output files are saved

## Smoke test plan

- dataset: HMDB51
- logger: TensorBoard
- epochs: 1–3
- goal: run the pipeline without errors


## Dataset understanding

- The codebase supports UCF101 and HMDB51.
- The dataset uses video files with annotation csv files.
- Each annotation line is: `path/to/video label`
- Video decoding is done on the fly with decord.

## Weight understanding

- Public pretrained checkpoints are listed in MODEL_ZOO.md.
- ViT-B is the most practical starting point for my project.

## Current plan

- smoke test on HMDB51
- later add UCF101 baseline if needed
- use single-GPU AutoDL instead of the official multi-node setup

## Script understanding

- The repo provides dataset-specific scripts under `scripts/`.
- The UCF101 fine-tuning template is available.
- The template uses `run_class_finetuning.py`.
- Key default settings in the UCF101 script:
  - `num_frames=16`
  - `sampling_rate=4`
  - `test_num_segment=5`
  - `test_num_crop=3`

## UCF101 official finetune template

- script path: `scripts/ucf101/.../finetune`
- entry: `run_class_finetuning.py`
- dataset: `UCF101`
- classes: `101`
- default frames: `16`
- default sampling rate: `4`
- default test segments: `5`
- default test crops: `3`

## My understanding

- this is an official reference script for UCF101 fine-tuning
- the original script is for multi-GPU distributed training
- later I need to simplify it for single-GPU AutoDL training
