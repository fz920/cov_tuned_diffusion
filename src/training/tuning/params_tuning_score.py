
import argparse
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Project imports (kept identical to your originals)
from cov_tuned_diffusion import ScoreNet, load_dataset, load_target_dist
from cov_tuned_diffusion.utils.path_config import (
    get_config_path,
    get_model_checkpoint_path,
    get_params_checkpoint_path,
    get_sample_path,
    get_figure_path,
    FIGURES_DIR,
    CHECKPOINTS_DIR,
)
import yaml

# ----------------------------
# Time-step utilities
# ----------------------------

def initialize_time_steps_params(num_steps: int, device: torch.device, T: float = 80.0, eps: float = 0.002) -> torch.Tensor:
    """
    Initialize the unconstrained parameters (logits) for the interior time steps.
    This reproduces your schedule parameterization:
        mu_n in (0,1) via sigmoid(logit), and
        t_{n-1} = eps + mu_{n-2} * (t_n - eps), for n = N-1,...,2

    Returns:
        mus_unconstrained: (num_steps-2,) tensor (logits)
    """
    r = (T / eps) ** (1 / num_steps)
    mus = torch.zeros(num_steps - 2)
    for n in range(2, num_steps):
        mus[n - 2] = (r ** (n - 1) - 1) / (r ** n - 1)
    mus = torch.logit(mus)  # logits
    return mus.to(device)


def get_time_steps(num_steps: int, mus: torch.Tensor, device: torch.device, eps: float = 0.002, T: float = 80.0) -> torch.Tensor:
    """
    Construct an increasing vector of time steps t_0=eps < t_1 < ... < t_{N-1}=T
    from interior mixing coefficients 'mus' in (0,1).

    Args:
        num_steps: total number of steps (>= 2)
        mus:       length (num_steps-2,) in (0,1)
        device:    torch device
        eps, T:    endpoints

    Returns:
        time_steps: (num_steps,) tensor
    """
    time_steps = torch.zeros(num_steps, device=device)
    time_steps[0], time_steps[-1] = eps, T
    for n in range(num_steps - 1, 1, -1):
        tn = time_steps[n]
        tn1 = (tn - eps) * mus[n - 2] + eps
        time_steps[n - 1] = tn1
    return time_steps


# ----------------------------
# Covariance builders
# ----------------------------

@dataclass
class CovBuildInputs:
    dataset: str
    n_dim: int
    n_particle: int
    time_steps: torch.Tensor               # (N,)
    proj_mat: torch.Tensor                 # (n_dim*n_particle, n_dim*(n_particle-1)) center-of-gravity projector

    # Generic "full" case (non-ALDP): rank-1+isotropic scalar family per step
    sigma2: Optional[torch.Tensor] = None  # (N-1,)
    lam: Optional[torch.Tensor] = None     # (N-1,)

    # ALDP-specific
    labels: Optional[torch.Tensor] = None  # (m,)
    atom_mat: Optional[torch.Tensor] = None# (N-1, num_types) for 'diag' or (N-1, num_types, num_types) for 'full'
    cov_form: Optional[Literal["diag","full"]] = None
    gammas: Optional[torch.Tensor] = None  # (N-1,) only when cov_form="full"


def build_covariance_matrices(inputs: CovBuildInputs) -> torch.Tensor:
    """
    Unified covariance builder covering:
      - Generic non-ALDP "full" case: B = (sigma2 - lam) I + lam 11^T
      - ALDP diag:                    B = diag( atom_mat[n][labels] )
      - ALDP full:                    B = atom_mat[n][labels, labels] + gamma_n^2 I

    Returns:
        cov_mat_all: (N-1, d_sub, d_sub) where d_sub = n_dim*(n_particle-1)
                     These are the *proposal* covariances multiplied by the DDPM scalar
                     (tn^2 / tn1^2 * (tn1^2 - tn^2)), ready to be fed into your low-rank samplers.
    """
    device = inputs.time_steps.device
    dtype = inputs.time_steps.dtype

    N = inputs.time_steps.numel()
    m = inputs.n_particle
    n = inputs.n_dim
    d_sub = n * (m - 1)

    I_n = torch.eye(n, device=device, dtype=dtype)
    cov_matrices = torch.zeros(N - 1, d_sub, d_sub, device=device, dtype=dtype)

    # ddpm scalar per step
    sigma2_ddpm = torch.zeros(N - 1, 1, 1, device=device, dtype=dtype)

    for k in range(N - 1):
        t_k, t_k1 = inputs.time_steps[k], inputs.time_steps[k + 1]

        if inputs.dataset != "aldp":
            assert inputs.sigma2 is not None and inputs.lam is not None, \
                "sigma2 and lam must be provided for non-ALDP datasets."
            sigma2_k = inputs.sigma2[k]
            lam_k    = inputs.lam[k]
            # B = (sigma2 - lam) I_m + lam * 11^T
            B = (sigma2_k - lam_k) * torch.eye(m, device=device, dtype=dtype) \
                + lam_k * torch.ones(m, m, device=device, dtype=dtype)
            cov_n = torch.kron(B, I_n)

        else:
            # ALDP branch
            assert inputs.cov_form in {"diag", "full"}, "cov_form must be 'diag' or 'full' for ALDP."
            assert inputs.labels is not None and inputs.atom_mat is not None, "labels and atom_mat are required for ALDP."

            if inputs.cov_form == "diag":
                # atom_mat[k]: (num_types,); pick per-atom variance
                diag = inputs.atom_mat[k][inputs.labels]      # (m,)
                B = torch.diag(diag)
                cov_n = torch.kron(B, I_n)

            else:  # "full"
                assert inputs.gammas is not None, "gammas must be provided for ALDP cov_form='full'."
                # atom_mat[k]: (num_types, num_types) symmetric base matrix
                full_B = inputs.atom_mat[k][inputs.labels[:, None], inputs.labels[None, :]] \
                         + (inputs.gammas[k] ** 2) * torch.eye(m, device=device, dtype=dtype)
                cov_n = torch.kron(full_B, I_n)

        # Project to zero-CG subspace and symmetrize
        low_rank = inputs.proj_mat.T @ cov_n @ inputs.proj_mat
        cov_matrices[k] = 0.5 * (low_rank + low_rank.T)

        # DDPM pre-factor
        sigma2_ddpm[k] = (t_k ** 2) / (t_k1 ** 2) * (t_k1 ** 2 - t_k ** 2)

    cov_mat_all = cov_matrices * sigma2_ddpm
    return cov_mat_all


# ----------------------------
# Training / tuning loop (unified)
# ----------------------------

def _to_positive(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Softplus stabilization for positive parameters."""
    return F.softplus(x) + eps


def _bounded_01(x: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    """Stable sigmoid within [lo, hi]."""
    return torch.sigmoid(x) * (hi - lo) + lo


def tune_params(args) -> None:
    """
    Unified front-end:
      - Loads model & data
      - Builds parameter tensors according to mode
      - Runs a simple optimizer loop maximizing forward ESS
      - Saves checkpoints/logs using your existing path helpers
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # --------- Load model & target ----------
    # Configs
    with open(get_config_path(args.dataset, str(args.model_index)), "r") as f:
        model_config = yaml.safe_load(f)

    score_model = ScoreNet(dataset=args.dataset, net=args.net, model_config=model_config, device=str(device))
    score_model = score_model.to(device)
    score_model.eval()  # tuning hyperparameters, model is fixed

    # Dataset (for labels if ALDP) & target distribution (for log_prob)
    _, true_target = load_target_dist(args.dataset, device=device)
    # time-step endpoints from model (kept aligned with your class)
    eps, T = score_model.eps, score_model.T

    # --------- Initialize parameters ----------
    # nus (reverse-variance multipliers): positive
    nus_unconstrained = torch.nn.Parameter(torch.zeros(args.num_steps - 1, device=device))
    # optional time step parameters
    mus_unconstrained = torch.nn.Parameter(initialize_time_steps_params(args.num_steps, device=device, T=T, eps=eps))
    params = [nus_unconstrained]
    if args.tune_time_steps:
        params.append(mus_unconstrained)

    # Covariance-specific params
    m, n = score_model._n_particles, score_model._n_dimension
    # Build a fixed full->subspace linear map (mn x n(m-1)) that drops center-of-mass.
    # E ∈ R^{m x (m-1)}: columns span {e_i - e_m}, i=1..m-1
    E = torch.zeros(m, m-1, device=device)
    E[:m-1, :m-1] = torch.eye(m-1, device=device)
    E[m-1, :] = -1.0
    proj_mat = torch.kron(E, torch.eye(n, device=device))  # (mn, n(m-1))

    cov_params = {}
    if args.mode == "full_generic":
        # Per-step sigma2 > 0 and 0 <= lam <= sigma2
        cov_params["sigma2_raw"] = torch.nn.Parameter(torch.zeros(args.num_steps - 1, device=device))
        cov_params["lam_raw"]    = torch.nn.Parameter(torch.zeros(args.num_steps - 1, device=device))
        params += [cov_params["sigma2_raw"], cov_params["lam_raw"]]
    elif args.mode == "aldp":
        # labels from the model if available
        assert hasattr(score_model, "atom_type_labels"), "ALDP requires atom_type_labels on the model."
        labels = score_model.atom_type_labels.to(device)
        num_types = int(labels.max().item() + 1)
        cov_params["labels"] = labels
        cov_params["cov_form"] = args.cov_form

        if args.cov_form == "diag":
            cov_params["atom_raw"] = torch.nn.Parameter(torch.zeros(args.num_steps - 1, num_types, device=device))
            params += [cov_params["atom_raw"]]
        elif args.cov_form == "full":
            cov_params["atom_raw"] = torch.nn.Parameter(torch.zeros(args.num_steps - 1, num_types, num_types, device=device))
            cov_params["gamma_raw"] = torch.nn.Parameter(torch.zeros(args.num_steps - 1, device=device))
            params += [cov_params["atom_raw"], cov_params["gamma_raw"]]
        else:
            raise ValueError("For ALDP, cov_form must be 'diag' or 'full'.")

    # --------- Optimizer ----------
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    # --------- Checkpoint paths ----------
    param_ckpt = get_params_checkpoint_path(args.dataset, str(args.model_index), str(args.params_index))
    Path(param_ckpt).parent.mkdir(parents=True, exist_ok=True)

    log_path = Path(CHECKPOINTS_DIR) / args.dataset / "logs"
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f"tuning__{args.mode}__m{args.model_index}__p{args.params_index}.log"

    # --------- Optimization loop ----------
    for epoch in tqdm(range(args.num_epochs), desc=f"Tuning ({args.mode})"):
        optimizer.zero_grad()

        # build time steps
        if args.tune_time_steps:
            mus = _bounded_01(mus_unconstrained)
            time_steps = get_time_steps(args.num_steps, mus, device=device, eps=eps, T=T)
        else:
            # use geometric schedule as default
            time_steps = torch.tensor(np.geomspace(eps, T, num=args.num_steps), device=device, dtype=torch.float32)

        # nus in (0, +inf); we multiply the base variance by nus
        nus = _to_positive(nus_unconstrained)

        # Build covariances if needed
        cov_mat_all = None
        if args.mode == "full_generic":
            sigma2 = _to_positive(cov_params["sigma2_raw"])
            lam    = _bounded_01(cov_params["lam_raw"], 0.0, 1.0) * sigma2  # lam \in [0, sigma2]
            cov_mat_all = build_covariance_matrices(CovBuildInputs(
                dataset=args.dataset,
                n_dim=n,
                n_particle=m,
                time_steps=time_steps,
                proj_mat=proj_mat,
                sigma2=sigma2,
                lam=lam,
            ))
        elif args.mode == "aldp":
            if args.cov_form == "diag":
                atom_mat = _to_positive(cov_params["atom_raw"])
                cov_mat_all = build_covariance_matrices(CovBuildInputs(
                    dataset="aldp",
                    n_dim=n,
                    n_particle=m,
                    time_steps=time_steps,
                    proj_mat=proj_mat,
                    labels=cov_params["labels"],
                    atom_mat=atom_mat,
                    cov_form="diag",
                ))
            else:  # full
                # symmetrize per-step base matrices and make them PSD-ish by softplus on diag
                raw = cov_params["atom_raw"]
                atom_sym = 0.5 * (raw + raw.transpose(-1, -2))
                # ensure positive diagonals
                diag = torch.diagonal(atom_sym, dim1=-2, dim2=-1)
                atom_sym = atom_sym.clone()
                for k in range(atom_sym.shape[0]):
                    atom_sym[k].diagonal().copy_(F.softplus(diag[k]))
                gammas = _to_positive(cov_params["gamma_raw"])
                cov_mat_all = build_covariance_matrices(CovBuildInputs(
                    dataset="aldp",
                    n_dim=n,
                    n_particle=m,
                    time_steps=time_steps,
                    proj_mat=proj_mat,
                    labels=cov_params["labels"],
                    atom_mat=atom_sym,
                    cov_form="full",
                    gammas=gammas,
                ))

        # ---- draw a batch of x0 and compute log_prob ----
        # For tuning we only need a reference batch to estimate forward ESS.
        # We keep it simple here; project-specific loaders can replace this with your canonical sampler.
        # Using the target distribution for log_prob and sampling x0 through the dataset if available:
        # (Assumes load_dataset returns a DataLoader or (train, val, test). Adjust if needed.)
        dataset = load_dataset(args.dataset, split="train", device=device)
        if hasattr(dataset, "__iter__"):
            try:
                batch = next(iter(dataset))
                x0 = batch.to(device)
            except Exception:
                # fallback: if dataset is not directly iterable into tensors, try calling .dataset[0]
                if hasattr(dataset, "dataset"):
                    x0 = dataset.dataset[0][0].to(device)
                    x0 = x0.unsqueeze(0).repeat(args.num_samples, 1, 1)
                else:
                    # As a last resort, sample standard Gaussian and rely on target log_prob (may mismatch)
                    x0 = torch.randn(args.num_samples, m, n, device=device)
        else:
            # fallback: random batch
            x0 = torch.randn(args.num_samples, m, n, device=device)

        # If the project requires removing CoG and/or scaling before evaluating log_prob, adjust here:
        x0 = x0[:args.num_samples]
        log_prob_x0 = true_target.log_prob(x0 / score_model.normalizing_constant)

        # ---- compute forward ESS with model adapters ----
        def _estimate_forward_ess_scalar():
            if hasattr(score_model, "est_forward_ess"):
                _, ess, _ = score_model.est_forward_ess(
                    x0, log_prob_x0, num_steps=args.num_steps, time_steps=time_steps,
                    nus=nus, progress_bar=False, alpha=args.alpha, tune_time_steps=args.tune_time_steps
                )
                return ess
            elif hasattr(score_model, "estimate_forward_ess"):
                _, ess, _ = score_model.estimate_forward_ess(
                    x0, log_prob_x0, num_steps=args.num_steps, time_steps=time_steps,
                    proposal="scalar", nus=nus, progress_bar=False, alpha=args.alpha, tune_time_steps=args.tune_time_steps
                )
                return ess
            else:
                raise RuntimeError("ScoreNet missing forward ESS method.")

        def _estimate_forward_ess_fullcov(cov):
            if cov is None:
                return _estimate_forward_ess_scalar()
            if hasattr(score_model, "forward_ess_low_rank"):
                _, ess, _ = score_model.forward_ess_low_rank(
                    x0, log_prob_x0, num_steps=args.num_steps, time_steps=time_steps,
                    cov_mat_all=cov, progress_bar=False, alpha=args.alpha
                )
                return ess
            elif hasattr(score_model, "estimate_forward_ess"):
                _, ess, _ = score_model.estimate_forward_ess(
                    x0, log_prob_x0, num_steps=args.num_steps, time_steps=time_steps,
                    proposal="fullcov", cov_mats=cov, progress_bar=False, alpha=args.alpha
                )
                return ess
            else:
                raise RuntimeError("ScoreNet missing forward ESS method.")

        forward_ess = _estimate_forward_ess_fullcov(cov_mat_all)
        loss = -forward_ess  # maximize ESS

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        if epoch % args.log_every == 0 or epoch == args.num_epochs - 1:
            with open(log_file, "a") as f:
                f.write(f"epoch={epoch} forward_ess={forward_ess.item():.6f}\n")

        # Save lightweight checkpoint
        if epoch % args.save_freq == 0 or epoch == args.num_epochs - 1:
            state = {
                "epoch": epoch,
                "nus_unconstrained": nus_unconstrained.detach().cpu(),
                "nus": _to_positive(nus_unconstrained).detach().cpu(),
                "tune_time_steps": args.tune_time_steps,
                "alpha": args.alpha,
                "time_steps": time_steps.detach().cpu(),
                "mode": args.mode,
            }
            if args.tune_time_steps:
                state["mus_unconstrained"] = mus_unconstrained.detach().cpu()
                state["mus"] = _bounded_01(mus_unconstrained).detach().cpu()
            if cov_mat_all is not None:
                state["cov_mat_all"] = cov_mat_all.detach().cpu()
            if args.mode == "full_generic":
                state["sigma2_raw"] = cov_params["sigma2_raw"].detach().cpu()
                state["lam_raw"]    = cov_params["lam_raw"].detach().cpu()
            if args.mode == "aldp":
                state["labels"] = cov_params["labels"].detach().cpu()
                state["cov_form"] = args.cov_form
                state["atom_raw"] = cov_params["atom_raw"].detach().cpu()
                if args.cov_form == "full":
                    state["gamma_raw"] = cov_params["gamma_raw"].detach().cpu()

            torch.save(state, param_ckpt)

    print("Finished tuning parameters.")


# ----------------------------
# CLI
# ----------------------------

def make_argparser():
    p = argparse.ArgumentParser(description="Unified parameter tuning (score/full/ALDP).")
    # generic
    p.add_argument("--dataset", type=str, default="aldp")
    p.add_argument("--net", type=str, default="egnn")
    p.add_argument("--model_index", type=int, default=0)
    p.add_argument("--params_index", type=int, default=0)

    # tuning knobs
    p.add_argument("--mode", type=str, choices=["score","full_generic","aldp"], default="score",
                   help="Which family to tune: scalar score params, generic full-cov, or ALDP-specific.")
    p.add_argument("--tune_time_steps", action="store_true", default=False)

    # optimization
    p.add_argument("--num_epochs", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--num_samples", type=int, default=512)
    p.add_argument("--save_freq", type=int, default=10)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--num_steps", type=int, default=10)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--cpu", action="store_true", help="Force CPU")

    # ALDP-only
    p.add_argument("--cov_form", type=str, choices=["diag","full"], default="diag",
                   help="ALDP covariance structure")

    return p


def main():
    args = make_argparser().parse_args()
    tune_params(args)


if __name__ == "__main__":
    main()
