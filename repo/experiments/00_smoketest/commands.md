# 00_smoketest commands

## Future steps on AutoDL

1. connect to the server with SSH
2. clone my project
3. clone the official VideoMAE repo
4. set up the environment
5. prepare the dataset
6. download pretrained weights
7. run 1–3 epochs
8. run evaluation
9. start TensorBoard
10. save screenshots and results

## Reference script

- official reference: `third_party/VideoMAE/scripts/ucf101/.../finetune`

## Reference

- official UCF101 fine-tuning script found in `third_party/VideoMAE/scripts/ucf101/.../finetune`
- later this script will be simplified for single-GPU smoke test

## Planned AutoDL paths

- project root: `/root/cv-research-videomae`
- official repo: `/root/cv-research-videomae/third_party/VideoMAE`
- dataset root: `/root/autodl-tmp/datasets`
- pretrained weights: `/root/autodl-tmp/pretrained`
- logs and outputs: `/root/autodl-tmp/runs`

## Planned smoke test setup

- dataset: HMDB51
- pretrained model: VideoMAE ViT-B
- num_frames: 16
- sampling_rate: 4
- epochs: 1–3
- logger: TensorBoard
- goal: run training, evaluation, and save one curve screenshot
