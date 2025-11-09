# Molecular Systems Pipeline

This folder hosts the end-to-end workflow used for ALDP, LJ13/LJ55, DW4, and related molecular systems.

All code reads/writes through `cov_tuned_diffusion/utils/path_config.py`. By default datasets live in `checkpoints/dataset/` (see `get_dataset_path`) and checkpoints land in `checkpoints/model_checkpoints/` or `checkpoints/params_checkpoints/`. Override the root via `export COV_TUNED_BASE=/path/to/storage` if needed.

## 0. Prepare data and configs

1. Place the raw datasets in the locations expected by `get_dataset_path` (e.g., `checkpoints/dataset/aldp_train.h5`, `dw4_samples.npy`, etc.).

The following datasets are used in our experiments:

### DW-4 (Double Well 4-particle system)
- **Source**: [OSF Repository](https://osf.io/srqg7/?view_only=28deeba0845546fb96d1b2f355db0da5)
- **Citation**: [1]
- **Filename**: Save as `dw4.pkl` in `checkpoints/dataset/`

### LJ-13 (Lennard-Jones 13-particle system)
- **Source**: [OSF Repository](https://osf.io/srqg7/?view_only=28deeba0845546fb96d1b2f355db0da5)
- **Citation**: [1]
- **Filename**: Save as `lj13.pkl` in `checkpoints/dataset/`

### Alanine Dipeptide
- **Source**: [Zenodo Repository](https://zenodo.org/records/6993124)
- **Citation**: [2]
- **Filename**: Save as `aldp.pkl` in `checkpoints/dataset/`

After downloading, place the datasets in the `checkpoints/dataset/` directory with the filenames specified above.

2. Ensure a model config exists at `src/configs/{dataset}_{net}_score_config.yaml`. The provided configs cover the paper’s settings.

## 1. Train the score model

```bash
python examples/molecules/score_training.py \
  --dataset aldp \
  --net egnn \
  --index 0 \
  --train_num_samples 4096 \
  --num_epochs 10000 \
  --save_freq 500
```

The script trains an EDM-style EGNN with EMA updates. Checkpoints plus optimizer state are written to `checkpoints/model_checkpoints/{dataset}/score/{net}_score_{index}.pth`, with a text log placed alongside the checkpoint.

## 2. Run basic diagnostics (optional)

```bash
python examples/molecules/score_test.py \
  --dataset aldp \
  --net egnn \
  --index 0 \
  --num_steps 500 \
  --num_samples 5000
```

This sampler compares ground-truth energies/distances against reverse-diffusion samples and reports the reverse ESS. Use `score_ode_test.py` for ODE-based checks and `measure_gflops.py` for FLOP/latency profiling when needed.

## 3. Tune scalar or covariance schedules

```bash
python examples/molecules/cov_tuning/tuning/params_tuning_score.py \
  --dataset aldp \
  --net egnn \
  --model_index 0 \
  --params_index 0 \
  --mode aldp \            # choices: score | full_generic | aldp
  --cov_form diag \        # aldp mode: diag | full
  --num_steps 40 \
  --num_epochs 5000 \
  --num_samples 512 \
  --tune_time_steps
```

Key modes:
- `score`: tunes scalar ν multipliers only (saved under `params_checkpoints_model` when `--cov_form` is unused).
- `full_generic`: tunes dataset-agnostic low-rank covariances.
- `aldp`: exposes molecule-specific diagonal or block-full covariances.

Checkpoints go to `checkpoints/params_checkpoints*/{dataset}/` according to the options you pass (`diag`, `tune_time_steps`, etc.). Logs land in `checkpoints/ess_log/{dataset}/`.

## 4. Sample and compute ESS with tuned parameters

```bash
python examples/molecules/sample_score.py \
  --dataset aldp \
  --net egnn \
  --model_index 0 \
  --params_index 0 \
  --num_steps 100 \
  --num_samples 5000 \
  --sample_num_times 5 \
  --cov_forms ddpm isotropic model full diag \
  --sampler ddpm \
  --tune_time_steps
```

What happens:
- Loads the requested ScoreNet checkpoint plus the tuned covariance/time-step files via `get_params_checkpoint_path`.
- Draws forward samples from the dataset (or Gaussian fallback), computes forward log-weights, and runs reverse DDPM/DDIM samplers.
- Writes per-covariance pickles to `checkpoints/samples/{dataset}/forward|backward/` and a log to `checkpoints/samples/{dataset}/logs/`.

Each batch reports forward and reverse ESS percentages in the log file. Increase `--sample_num_times` or `--num_samples` to accumulate more statistics; resume runs with `--continue_sampling`.

## 5. Analyze the results

- Forward/backward pickles contain `log_weights` and (for backward) `samples`, so you can build ESS summaries or downstream observables.
- Figures such as Ramachandran plots, reweighted histograms, and α-sweeps are generated through the scripts in `src/analysis/plotting/` (see the top-level repo README for exact commands).

---

**Tips**

- Matching `--model_index`, `--params_index`, and `--sample_index` across steps keeps checkpoints organized.
- Use `--cpu` flags for tuning/sampling dry runs on smaller subsets before launching large GPU jobs.
- When combining custom covariance families, inspect the saved checkpoints (they include `time_steps`, `nus`, and `cov_mat_all`) or reuse `examples/molecules/cov_tuning/schedules.py` utilities for additional analysis.
