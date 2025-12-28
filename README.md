# RM -RF

This repository contains the data used in the experiments:

- The file **`holdout_dataset.zip`** includes three `.jsonl` files — one for each target type:  
  `binary_type`, `float_type`, and `reverse_binary_type`.

- The file **`train_data.zip`** contains six `.jsonl` files: training (`train`) and validation (`val`) splits for each of the three target types.

- The **`validation_subset`** directory contains `.toml` files — one per sample.

- The **`slurm_scripts_examples`** directory contains example `.sh` scripts for:
  - Full fine-tuning,
  - Fine-tuning with LoRA,
  - Inference.

- The **`prompts`** directory includes:
  - A prompt template used to generate test-breaking inputs,
  - Descriptions of errors that were synthetically generated using an LLM.