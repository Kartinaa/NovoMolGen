## Pretraining and Fine-Tuning for Molecular Language Models

This directory contains training scripts for both pretraining and fine-tuning of molecular language models.

- **Pretraining** uses the standard [HuggingFace `Trainer`](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py).
- **Fine-tuning** extends the [TRL library](https://github.com/huggingface/trl) and implements multiple reinforcement learning–based strategies:
  - **REINVENT-style fine-tuning** using a fixed prior and task-specific reward function.
  - **Augmented Hill-Climb (AHC)** with top-k molecule selection and experience replay.
  - **Supervised Fine-Tuning (SFT)** on top-reward molecules selected from pre-scored datasets.

Each fine-tuning strategy is implemented in its own trainer class:
- `reinvent_trainer.py`
- `augment_hc_trainer.py`
- `sft_trainer.py`

For method details, refer to our paper and the docstrings provided in each trainer implementation.