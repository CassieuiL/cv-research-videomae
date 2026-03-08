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
