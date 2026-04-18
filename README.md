# PADDock

**P**hysical-**A**lignment **D**iffusion **Docking** via RL Post-training

PADDock is a reinforcement learning (RL) post-training framework for diffusion-based molecular docking models. Built on top of models such as DiffDock, it is designed to improve **physical plausibility** in terms of binding energy while preserving **geometric accuracy** in terms of RMSD.

## Overview

<div align="center">
  <img src="visualizations/overview.png" width="85%" alt="PADDock overview">
</div>

PADDock targets the molecular docking setting, where geometric correctness and physical binding quality often need to be balanced carefully. The core workflow consists of:

1. Data preparation and optional embeddings
2. Optional cache pre-generation
3. RL post-training
4. Inference and sampling with the trained model

## Environment Setup

We recommend creating the environment from the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate PADDock
```

## Project Structure

- `train_ddpo.py`: main entry point for RL post-training
- `inference_rl.py`: main entry point for inference and sampling
- `pregenerate_cache.py`: optional cache pre-generation before training or inference
- `datasets/`: scripts related to data and feature processing
- `workdir/`: directory for checkpoints and model weights
- `results/`: directory for training and inference outputs

## Data Preparation

Prepare the following components according to your experimental setup:

1. Dataset files, for example under `data/`
2. Data split files for train/validation/test
3. Protein and ligand inputs for inference, either as single examples or in CSV format

If cached features or preprocessing artifacts are used, please ensure that all paths are consistent with your local configuration.

## Optional: Cache Pre-generation

```bash
python pregenerate_cache.py
```

Running this step before training is recommended, as it can reduce waiting time during subsequent stages. Before execution, ensure that the relevant configuration and model files under `workdir/` are available.

## Training

Single-GPU training:

```bash
python train_ddpo.py
```

Example multi-GPU training:

```bash
torchrun --nproc_per_node=4 train_ddpo.py
```

Typical training outputs include:

- intermediate checkpoints, for example `rl_model_epoch_*.pt`
- training statistics, for example `training_stats.csv`
- additional experiment artifacts saved to the output directory configured in `train_ddpo.py`

## Inference

Batch inference with CSV input:

```bash
python inference_rl.py --protein_ligand_csv data/testset_csv.csv --out_dir results/rl_inference
```

Single-complex inference example:

```bash
python inference_rl.py \
  --complex_name 1a0q \
  --protein_path data/1a0q/1a0q_protein_processed.pdb \
  --ligand_description "CCCCC(NC(=O)CCC(=O)O)P(=O)(O)OC1=CC=CC=C1" \
  --out_dir results/rl_single
```

Common arguments:

- `--out_dir`: output directory
- `--protein_ligand_csv`: CSV file for batch inference
- `--complex_name` / `--protein_path` / `--ligand_description`: inputs for a single complex
- `--samples_per_complex`: number of samples generated per complex
- `--batch_size`: inference batch size
- `--inference_steps`: number of denoising steps

Inference results are written under `--out_dir`, typically in per-complex subdirectories containing generated `sample_*.sdf` files.

## Minimal Workflow

```bash
# 1. Optional: pre-generate cache
python pregenerate_cache.py

# 2. Train
python train_ddpo.py

# 3. Run inference
python inference_rl.py --protein_ligand_csv data/testset_csv.csv --out_dir results/rl_inference
```

## Acknowledgements

This project builds upon the excellent open-source work of **DiffDock** by G. Corso et al.

We thank the original authors for releasing their code and models to the community.

## License

This project is released under the MIT License.

Parts of this repository include modifications based on DiffDock. The original DiffDock project is also distributed under the MIT License.
