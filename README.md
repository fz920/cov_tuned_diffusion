# Variance-Tuned Diffusion Importance Sampling

A framework for improved score-based generative model sampling via tuned covariance.

## Overview

This repository implements our proposed sampling methods for score-based generative models, focusing on improving the effective sample size (ESS) and sampling efficiency.

## Features

- Score-based generative model training with various architectures (EGNN, etc.)
- Advanced sampling techniques with improved ESS
- Support for multiple datasets including DW-4, LJ-13 and Alanine Dipeptide
- Parameter tuning for optimized sampling performance
- Visualization tools for analyzing results and sampling quality

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd cov_tuned_diffusion

# Install required dependencies
pip install -r requirements.txt
```

## Directory Structure

- `model/`: Neural network architectures and model components
  - `diffusion.py`: Score-based diffusion models
  - `egnn.py`: E(n) Equivariant Graph Neural Networks
  - `flows/`: Normalizing flow models
  - `configs/`: Model configuration files
- `target_dist/`: Target distribution implementations
- `utils/`: Utility functions and configurations
  - `path_config.py`: Central path configuration system
- `figures/`: Generated plots and visualizations
- Various scripts for training, sampling, and evaluation

## Path Configuration

The codebase uses a centralized path configuration system in `utils/path_config.py` to manage all file paths. This makes it easy to:

1. Configure all paths from a single location
2. Automatically create necessary directories
3. Maintain consistent path structures across all scripts

All data, model checkpoints, and output files are organized as follows:

```
checkpoints/
├── dataset/            # Dataset files
├── model_checkpoints/  # Trained model weights
├── params_checkpoints/ # Parameter tuning results
├── samples/            # Generated samples
└── ...
```

## Usage

### Training a Score Model

```bash
python score_training.py --dataset dw4 --net egnn --num_epochs 10000 --train_num_samples 2000
```

### Parameter Tuning

```bash
python params_tuning_score.py --dataset dw4 --net egnn --model_index 0 --num_steps 100
```

### Sampling

```bash
python sample_score.py --dataset dw4 --net egnn --params_index 0 --model_index 0 --num_samples 5000 --num_steps 100
```

### Evaluating Effective Sample Size (ESS)

```bash
python forward_ess_score.py --dataset aldp --net egnn --params_index 0 --model_index 0
```

### Visualizing Results

```bash
python plot_ess_score.py --dataset aldp --net egnn
python plot_reweighted_hist.py --dataset aldp --net egnn
```

## Citation

```
@inproceedings{anonymous2024efficient,
  title={Efficient and Unbiased Sampling from Boltzmann Distributions via Variance-Tuned Diffusion Models},
  author={Anonymous},
  booktitle={Submitted to NeurIPS 2025},
  year={2025}
}
```

## License

[MIT License](LICENSE)
