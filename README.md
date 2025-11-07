# Variance-Tuned Diffusion Importance Sampling

**Implementation for the TMLR paper**  
*Efficient and Unbiased Sampling from Boltzmann Distributions via Variance-Tuned Diffusion Models* (OpenReview: [https://openreview.net/forum?id=Jq2dcMCS5R](https://openreview.net/forum?id=Jq2dcMCS5R))

> **At a glance**
>
> 1. Train score models → tune noise schedules → sample with reweighting → evaluate ESS and downstream metrics.
> 2. All scripts use `cov_tuned_diffusion/utils/path_config.py`, so datasets, checkpoints, and figures land in predictable locations.

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
pip install -e .
```

---

## Reproducing Paper Figures

* **ESS comparisons (score models)**
  ```bash
  python -m cov_tuned_diffusion.analysis.plotting.plot_ess_score --dataset dw4 --net egnn \
      --model-index 0 --params-index-list 0 --num-steps-list 50 100
  ```
* **Reweighted histograms (Alanine dipeptide)**
  ```bash
  python -m cov_tuned_diffusion.analysis.plotting.plot_reweighted_hist --dataset aldp --net egnn
  ```
* **GMM benchmarks**
  ```bash
  python examples/gmm/compute_ess.py
  python examples/gmm/plot_ess.py
  python examples/gmm/plot_gmm_hist.py
  python examples/gmm/plot_var.py
  ```
* **α-sweep analyses (molecular systems)**
  ```bash
  python -m cov_tuned_diffusion.analysis.plotting.plot_klalpha_ess
  ```

Dataset names, checkpoint indices, and parameter IDs depend on the run; refer to `docs/structure.md` for naming conventions.

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
