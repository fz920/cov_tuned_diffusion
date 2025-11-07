# Variance-Tuned Diffusion Importance Sampling

Implementation for the TMLR paper *Efficient and Unbiased Sampling from Boltzmann Distributions via Variance-Tuned Diffusion Models* ([OpenReview](https://openreview.net/forum?id=Jq2dcMCS5R)).

> **Workflow:** train score models → tune covariances/time steps → sample with reweighting → report ESS & downstream metrics.  
> All scripts rely on `cov_tuned_diffusion/utils/path_config.py`, so datasets, checkpoints, and figures land in predictable folders (override with `COV_TUNED_BASE` if desired).

---

## Quick Start

```bash
git clone https://github.com/fz920/cov_tuned_diffusion.git
cd cov_tuned_diffusion
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## Experiment Guides

- **Molecular systems (ALDP, LJ13/55, DW4):** `examples/molecules/README.md`
- **Synthetic GMM benchmarks:** `examples/gmm/README.md`
- **Analysis/figure scripts:** `src/analysis/plotting/` (commands listed in each module)

Each README details the end-to-end pipeline (training → tuning → sampling → plotting) and the relevant CLI flags.

---

## Common Figure Commands

```bash
# Score-model ESS curves
python -m cov_tuned_diffusion.analysis.plotting.plot_ess_score \
  --dataset dw4 --net egnn --model-index 0 --params-index-list 0 --num-steps-list 50 100

# Reweighted histograms (ALDP)
python -m cov_tuned_diffusion.analysis.plotting.plot_reweighted_hist --dataset aldp --net egnn

# α-sweep analyses (molecules)
python -m cov_tuned_diffusion.analysis.plotting.plot_klalpha_ess
```

For GMM visuals, run the scripts documented in `examples/gmm/README.md`.

---

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
