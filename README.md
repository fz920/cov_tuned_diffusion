# Variance-Tuned Diffusion Importance Sampling

**Implementation for the TMLR paper**  
*Efficient and Unbiased Sampling from Boltzmann Distributions via Variance-Tuned Diffusion Models* (OpenReview: [https://openreview.net/forum?id=Jq2dcMCS5R](https://openreview.net/forum?id=Jq2dcMCS5R))

This repository delivers an end-to-end workflow for training score-based diffusion models, tuning variance or covariance schedules, and drawing unbiased importance-sampling estimates. It covers molecular systems (Alanine dipeptide, Lennard-Jones) and synthetic Gaussian mixture model (GMM) benchmarks, and includes plotting utilities that reproduce the results reported in the paper.

> **At a glance**
>
> 1. Train score models → tune noise schedules → sample with reweighting → evaluate ESS and downstream metrics.
> 2. All scripts use `utils/path_config.py`, so datasets, checkpoints, and figures land in predictable locations.

---

## Table of Contents

* [Features](#features)
* [Installation](#installation)
* [Repository Layout & Paths](#repository-layout--paths)
* [Quickstart](#quickstart)
* [Reproducing Paper Figures](#reproducing-paper-figures)
* [Data & Configuration Notes](#data--configuration-notes)
* [Outputs & Logging](#outputs--logging)
* [Contributing](#contributing)
* [Citation](#citation)
* [License](#license)
* [Troubleshooting](#troubleshooting)

---

## Features

* Unified pipeline for training, variance tuning, sampling with reweighting, and plotting.
* Centralized path management (`utils/path_config.py`) keeps experiments organized across machines.
* Ready-made experiment modules for molecular systems and synthetic GMM benchmarks.
* Reproducibility aids such as consistent directory structure and figure scripts.

For a walkthrough of the code and data flow, see `docs/structure.md`.

---

## Installation

```bash
# Clone the repository
git clone <your-repository-url>
cd cov_tuned_diffusion

# Create and activate a Python environment (Python ≥ 3.9)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

GPU acceleration is optional but recommended for training; exact package versions are listed in `requirements.txt`.

---

## Repository Layout & Paths

All scripts rely on the directory skeleton managed by `utils/path_config.py`. On first run, it creates:

```
checkpoints/
├── dataset/                 # Cached datasets
├── gmm2_checkpoints/        # GMM checkpoints, tuned params, ESS summaries
├── model_checkpoints/       # Score-model weights
├── params_checkpoints*/     # Tuned covariance/time-step parameters
└── samples*/                # Diffusion trajectories and final samples
figures/
└── ...                      # Plots and tables
```

`*` marks subfolders created per dataset or run. To store artifacts elsewhere, set `COV_TUNED_BASE=/path/to/workdir`. Defaults work for both local and HPC deployments.

---

## Quickstart

1. **Prepare data.** Export raw datasets to `checkpoints/dataset/`. For GMM experiments, place pretrained score checkpoints in `checkpoints/gmm2_checkpoints/model_checkpoints/` (see `docs/structure.md` for filenames).
2. **Train score models.**
   ```bash
   python score_training.py --dataset dw4 --net egnn \
       --num_epochs 10000 --train_num_samples 2000
   ```
   Models are saved via `get_model_checkpoint_path(...)`.
3. **Tune covariance/time-step schedules.**
   ```bash
   python tune_var/params_tuning_score_full.py \
       --dataset dw4 --net egnn --num_steps 100 --params_index 0
   ```
   GMM-specific tuning lives in `gmm2_experiments/tune_params_score.py`.
4. **Sample with tuned parameters.**
   ```bash
   python sample_score.py --dataset dw4 --net egnn \
       --model_index 0 --params_index 0 --num_steps 100 --num_samples 5000
   ```
5. **Evaluate and plot.**
   ```bash
   python plotting/plot_ess_score.py --dataset dw4 --net egnn \
       --model-index 0 --params-index-list 0 --num-steps-list 50 100

   python plotting/plot_reweighted_hist.py --dataset aldp --net egnn
   ```

Each script provides `--help` for the complete argument list and writes outputs through the shared path helpers.

---

## Reproducing Paper Figures

* **ESS comparisons (score models)**
  ```bash
  python plotting/plot_ess_score.py --dataset dw4 --net egnn \
      --model-index 0 --params-index-list 0 --num-steps-list 50 100
  ```
* **Reweighted histograms (Alanine dipeptide)**
  ```bash
  python plotting/plot_reweighted_hist.py --dataset aldp --net egnn
  ```
* **GMM benchmarks**
  ```bash
  python gmm2_experiments/compute_ess.py
  python gmm2_experiments/plot_ess.py
  python gmm2_experiments/plot_gmm_hist.py
  python gmm2_experiments/plot_var.py
  ```
* **α-sweep analyses (molecular systems)**
  ```bash
  python plotting/plot_klalpha_ess.py
  ```

Dataset names, checkpoint indices, and parameter IDs depend on the run; refer to `docs/structure.md` for naming conventions.

---

## Data & Configuration Notes

* **Model configs:** stored in `model/configs/` and accessed via `get_config_path(dataset, net, type="score")`.
* **Alanine dipeptide:** utilities live under `target_dist/aldp_*`; a low-rank tuner is available in `tune_var/params_tuning_score_full_aldp.py`.
* **GMM benchmarks:** standalone score networks sit in `gmm2_experiments/score_model.py`; checkpoints, tuned parameters, and ESS summaries are under `checkpoints/gmm2_checkpoints/`.

---

## Outputs & Logging

* **Model weights:** `checkpoints/model_checkpoints/…`
* **Tuned schedules:** `checkpoints/params_checkpoints*/…`
* **Samples and trajectories:** `checkpoints/samples*/…`
* **Figures and tables:** `figures/…` (PNG/PDF)
* **Console logs:** stream to stdout; redirect if needed:
  ```bash
  python sample_score.py ... | tee logs/sample_dw4_egnn.txt
  ```

For large runs or HPC jobs, set `COV_TUNED_BASE` to a high-throughput filesystem such as `$SLURM_TMPDIR` or scratch space.

---

<!-- ## Contributing

Contributions are welcome. Please open an issue or pull request if you spot discrepancies with the paper or want to extend the experiment suite. Route filesystem interactions through `utils/path_config.py` to keep the layout consistent.

--- -->

## Citation

If you use this codebase, please cite:

```bibtex
@article{zhang2025efficient,
  title   = {Efficient and Unbiased Sampling from Boltzmann Distributions via Variance-Tuned Diffusion Models},
  author  = {Zhang, Fengzhe and Midgley, Laurence I. and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel},
  journal = {Transactions on Machine Learning Research (TMLR)},
  year    = {2025},
  url     = {https://openreview.net/forum?id=Jq2dcMCS5R}
}
```

The final camera-ready manuscript and supplementary material are available on the TMLR OpenReview page.

---

## License

This project is released under the [MIT License](LICENSE).

<!-- --- -->

<!-- ## Troubleshooting -->

<!-- * **Unexpected paths or missing files:** verify `COV_TUNED_BASE` (if set) and inspect the auto-generated `checkpoints/` and `figures/` folders.
* **Flag style confusion:** some scripts use `--flag_name`, others use `--flag-name`; follow the style shown in each script’s `--help`.
* **Dataset lookup errors:** confirm `checkpoints/dataset/` contains the expected exports (see `docs/structure.md`).
* **Long runtimes or out-of-memory:** lower `--num_steps`, batch sizes, or sample counts; use a GPU when available. -->
