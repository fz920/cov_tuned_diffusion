# Gaussian Mixture (GMM2) Pipeline

End-to-end recipe for the synthetic GMM benchmarks used in the paper. Every script in this folder writes to the locations defined in `cov_tuned_diffusion/utils/path_config.py`—by default `<repo>/../checkpoints/gmm2_*`. Set `COV_TUNED_BASE=/path/to/storage` if you want outputs elsewhere.

## 1. Train the score model

```bash
python examples/gmm/training.py \
  --input_dim 2 \
  --n_layers 7 \
  --hidden_size 512 \
  --train_num_samples 5000 \
  --num_epochs 50000 \
  --save_freq 1000
```

This script trains an EDM-style ScoreNet on synthetic data drawn from `gmm.create_gmm`. It saves EMA checkpoints to `checkpoints/gmm2_checkpoints/model_checkpoints/{input_dim}D_gmm2_score_ckpt_...pth`.

## 2. Sanity-check the model (optional but recommended)

```bash
python examples/gmm/score_test.py \
  --input_dim 2 \
  --n_layers 7 \
  --hidden_size 512 \
  --num_steps 500 \
  --num_samples 10000
```

Produces histograms and scatter plots in `figures/gmm2/diagnostics/` plus an ESS estimate from reverse sampling. Use this to verify the checkpoint before moving on.

## 3. Tune covariance / time-step schedules

```bash
python examples/gmm/tune_params_score.py \
  --input_dim 2 \
  --cov_form isotropic \
  --params_index 0 \
  --num_steps 20 \
  --num_epochs 5000 \
  --alpha 2.0 \
  --rank -1
```

The tuner maximizes forward ESS over batches from the true GMM and stores checkpoints via `get_gmm2_params_checkpoint_path`, e.g. `checkpoints/gmm2_checkpoints/params_checkpoints/50D_gmm2_score_params_25steps_0_diagonal.pth`. Run the command per `(cov_form, num_steps, params_index)` you intend to evaluate.

## 4. Evaluate ESS with the tuned parameters

```bash
python examples/gmm/compute_ess.py \
  --input_dim 2 \
  --num_steps 20 \
  --num_samples 100000 \
  --params_index_list 0 \
  --rank -1
```

`compute_ess.py` loads the trained ScoreNet plus each requested covariance form (`ddpm`, `isotropic`, `diagonal`, `full`). It reports forward/reverse ESS percentages and saves a text summary to `checkpoints/gmm2_checkpoints/ess_checkpoints/`.

## 5. Plot ESS curves

```bash
python examples/gmm/plot_ess.py \
  --input_dims 2 \
  --num_steps_list 20 40 \
  --params_index_list 0 0 0 \
  --rank -1
```

If the summary file already exists, the plotter reuses it; otherwise it triggers `compute_ess.py`. The resulting multi-panel ESS figure is written under `figures/gmm2/ess/`.

## 6. Visualize reweighted histograms

```bash
python examples/gmm/plot_gmm_hist.py \
  --input_dims 2 \
  --cov_form isotropic \
  --rank -1 \
  --num_steps 20 \
  --num_samples 10000 \
  --sample_index 0
```

Generates `figures/gmm2/hist/*reweight*.pdf`, comparing ground-truth log-probabilities against raw and reweighted diffusion samples. The script automatically reuses cached samples in `checkpoints/gmm2_checkpoints/sample_checkpoints/backward/` when available.

## 7. Inspect tuned schedules / ν parameters

```bash
python examples/gmm/plot_var.py \
  --input_dim 50 \
  --num_steps 25 \
  --params_index 0 \
  --cov_form isotropic
```

This produces a trend plot of the learned per-step multipliers (and, when available, tuned time steps) under `figures/gmm2/var/`. Use it to compare tuned vs. geometric schedules or multiple covariance families.

---

**Tips**

- Reuse `--params_index`/`--sample_index` to keep checkpoints organized per sweep.
- Increase `--num_samples` in `compute_ess.py` for smoother statistics; plotting scripts expect the same indices/rank you used during tuning.
- Large-dimensional runs benefit from setting `CUDA_VISIBLE_DEVICES` and running tuning/evaluation on separate GPUs to avoid checkpoint contention.
