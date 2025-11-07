"""Variance/covariance schedule helpers used during tuning."""

from __future__ import annotations

import numpy as np
import torch

from ..models.utils import construct_R, sample_center_gravity_zero_gaussian


def get_constrained_params(mus_unconstrained, etas_unconstrained, gammas_unconstrained, nus_unconstrained):
    mus = torch.sigmoid(mus_unconstrained)
    etas = torch.sigmoid(etas_unconstrained)
    gammas = torch.sigmoid(gammas_unconstrained)
    nus = torch.sigmoid(nus_unconstrained)
    return mus, etas, gammas, nus


def initialize_parameters(num_steps, device: str = "cuda", T: float = 80.0, eps: float = 0.0002):
    etas = torch.logit(torch.ones(num_steps - 1) * 0.1).to(device)

    r = (T / eps) ** (1 / num_steps)
    mus = torch.zeros(num_steps - 2)
    for n in range(2, num_steps):
        mus[n - 2] = (r ** (n - 1) - 1) / (r ** n - 1)
    mus = torch.logit(mus).to(device)

    gammas = torch.logit(torch.ones(num_steps - 1) * 0.9).to(device)
    nus = torch.logit(torch.ones(num_steps - 1) * 0.9).to(device)

    etas = torch.nn.Parameter(etas)
    mus = torch.nn.Parameter(mus)
    gammas = torch.nn.Parameter(gammas)
    nus = torch.nn.Parameter(nus)
    return etas, mus, gammas, nus


def initialize_parameters_abl(num_steps, device: str = "cuda", T: float = 80.0, eps: float = 0.0002, remove_params: str = "etas"):
    if remove_params == "etas":
        etas = torch.logit(torch.ones(num_steps - 1) * 0.0).to(device)
    else:
        etas = torch.logit(torch.ones(num_steps - 1) * 0.1).to(device)

    r = (T / eps) ** (1 / num_steps)
    mus = torch.zeros(num_steps - 2)
    for n in range(2, num_steps):
        mus[n - 2] = (r ** (n - 1) - 1) / (r ** n - 1)
    mus = torch.logit(mus).to(device)

    if remove_params == "gammas":
        gammas = torch.logit(torch.ones(num_steps - 1) * 1).to(device)
    else:
        gammas = torch.logit(torch.ones(num_steps - 1) * 0.9).to(device)

    if remove_params == "nus":
        nus = torch.logit(torch.ones(num_steps - 1) * 1).to(device)
    else:
        nus = torch.logit(torch.ones(num_steps - 1) * 0.9).to(device)

    if remove_params == "etas and gammas":
        etas = torch.logit(torch.ones(num_steps - 1) * 0.0).to(device)
        gammas = torch.logit(torch.ones(num_steps - 1) * 1).to(device)

    etas = torch.nn.Parameter(etas)
    mus = torch.nn.Parameter(mus)
    gammas = torch.nn.Parameter(gammas)
    nus = torch.nn.Parameter(nus)
    return etas, mus, gammas, nus


def get_time_steps(num_steps, etas, gammas, mus, eps: float = 0.0002, T: float = 80, device: str = "cuda"):
    assert len(etas) == num_steps - 1
    assert len(gammas) == num_steps - 1
    assert len(mus) == num_steps - 2
    time_steps = torch.zeros(num_steps).to(device)
    time_steps[0], time_steps[-1] = eps, T
    for n in range(num_steps - 1, 1, -1):
        tn = time_steps[n]
        tn1 = (tn - eps) * mus[n - 2] + eps
        time_steps[n - 1] = tn1

    target_time_points = torch.zeros(num_steps - 1,)
    proposal_time_points = torch.zeros(num_steps - 1,)

    for n in range(len(time_steps) - 1, 0, -1):
        tn = time_steps[n]
        tn1 = time_steps[n - 1]

        u1 = (tn - tn1) * etas[n - 1] + tn1
        target_time_points[n - 1] = u1

        u2 = (tn - tn1) * gammas[n - 1] + tn1
        proposal_time_points[n - 1] = u2

    return time_steps, target_time_points, proposal_time_points


def get_ocm_cov(x0, num_steps, num_samples_est_hess, score_model):
    proj_mat = construct_R(score_model._n_particles, score_model._n_dimension, device=score_model.device)
    eps, T = score_model.eps, score_model.T
    time_steps = torch.tensor(np.geomspace(eps, T, num_steps)).to(score_model.device)

    subspace_dim = score_model.subspace_dim
    ocm_cov = torch.zeros(num_steps - 1, subspace_dim, subspace_dim).to(score_model.device)
    for n in range(num_steps - 1, 0, -1):
        tn, tn1 = time_steps[n], time_steps[n - 1]

        x_tn = x0 + tn * sample_center_gravity_zero_gaussian(x0.shape, device=score_model.device)

        x0_hat = score_model(x_tn, tn * torch.ones(num_samples_est_hess, device=score_model.device))
        score_tn = -(x_tn - x0_hat) / (tn**2)

        score_tn = score_tn.view(num_samples_est_hess, -1)
        score_tn_proj = score_tn @ proj_mat

        hess_est = -torch.einsum("bi,bj->bij", score_tn_proj, score_tn_proj).mean(0)

        sigma = (tn**2 - tn1**2) ** 0.5

        ocm_cov[n - 1] = sigma**4 * hess_est + sigma**2 * torch.eye(subspace_dim, device=score_model.device)

        sigma2_ddpm = (tn1**2 * (tn**2 - tn1**2)) / tn**2
        ocm_cov[n - 1] = ocm_cov[n - 1] / sigma2_ddpm

        try:
            torch.linalg.cholesky(ocm_cov[n - 1])
        except RuntimeError:
            print("Cholesky failed at time step:", n)
            breakpoint()

    return ocm_cov


__all__ = [
    "get_constrained_params",
    "initialize_parameters",
    "initialize_parameters_abl",
    "get_time_steps",
    "get_ocm_cov",
]
