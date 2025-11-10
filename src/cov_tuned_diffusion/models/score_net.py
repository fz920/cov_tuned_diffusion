from typing import Literal, Optional, List

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from torch.func import vmap, jacrev, jvp

from .utils import assert_mean_zero, remove_mean, sample_center_gravity_zero_gaussian, center_gravity_zero_gaussian_log_likelihood, create_edges, cast_edges2batch, TimeEmbedding_Diffusion, center_of_gravity_gaussian_log_likelihood_full_cov, sample_center_of_gravity_gaussian_full_cov, compute_forward_ess
from .egnn import EGNN

class ScoreNet(nn.Module):
    def __init__(
        self,
        dataset='lj13',
        net='egnn',
        model_config=None,
        device='cuda'
    ):
        super().__init__()
        self.net_name = net
        self.device = device
        self.model_config = model_config
        self.dataset = dataset

        if net == 'egnn':
            self.model = EGNN(
                in_node_nf=model_config['time_embedding_dim'] + model_config['atom_type_embedding_dim'],
                in_edge_nf=2 if dataset == 'aldp' else 1,
                hidden_nf=model_config['hidden_nf'],
                act_fn=torch.nn.SiLU(),
                n_layers=model_config['n_layers'],
                recurrent=model_config['recurrent'],
                attention=model_config['attention'],
                tanh=model_config['tanh'],
                agg=model_config['agg'],
                norm_constant=1,  # normalization constant is changed to 1            
            )

        else:
            raise ValueError('Model not supported')

        if dataset == 'dw4':
            self.sigma_data = 1.8
            self._n_particles = 4
            self._n_dimension = 2

            self.eps = 0.002
        elif dataset == 'lj13':
            self.sigma_data = 0.68
            self._n_particles = 13
            self._n_dimension = 3

            self.eps = 0.002

        elif dataset == 'lj55':
            self.sigma_data = 1
            self._n_particles = 55
            self._n_dimension = 3

            self.eps = 0.002

        elif dataset == 'aldp':
            self.sigma_data = 0.168
            self._n_particles = 22
            self._n_dimension = 3

            atom_type_labels = [
                0,  # H 0
                1,  # C 1
                0,  # H 2
                0,  # H 3
                1,  # C 4
                2,  # O 5
                3,  # N 6
                0,  # H 7
                1,  # C 8
                0,  # H 9
                1,  # C 10
                0,  # H 11
                0,  # H 12
                0,  # H 13
                1,  # C 14
                2,  # O 15
                3,  # N 16
                0,  # H 17
                1,  # C 18
                0,  # H 19
                0,  # H 20
                0   # H 21
            ]

            self.atom_type_labels = torch.tensor(atom_type_labels, device=self.device)
            num_atom_types = len(set(atom_type_labels))
            self.atom_type_embedding_layer = nn.Embedding(num_atom_types, model_config['atom_type_embedding_dim'])

            # Define the bond list for alanine dipeptide
            self.bonds = [
                (0, 1), (1, 2), (1, 3), (1, 4),
                (4, 5), (4, 6), (6, 7), (6, 8),
                (8, 9), (8, 10), (10, 11), (10, 12),
                (10, 13), (8, 14), (14, 15), (14, 16),
                (16, 17), (16, 18), (18, 19), (18, 20),
                (18, 21)
            ]

            # Create adjacency matrix for bond information
            adj = torch.zeros((self._n_particles, self._n_particles), dtype=torch.float32, device=self.device)
            for i, j in self.bonds:
                adj[i, j] = 1.0
                adj[j, i] = 1.0  # Assuming undirected bonds
            self.adj = adj

            self.eps = 0.002

        else:
            raise ValueError('Dataset not supported')

        self.subspace_dim = (self._n_particles - 1) * self._n_dimension
        self.full_dim = self._n_particles * self._n_dimension

        self.T = 80.0

        self.edges = create_edges(self._n_particles)
        self._edges_dict = {}
        # self.edges = self._create_edges()
        # self._edges_dict = {}

        self.time_embedding = TimeEmbedding_Diffusion(model_config['time_embedding_dim']).to(self.device)

        self.normalizing_constant = 10.0 if dataset == 'aldp' else 1.0

    def forward(self, xt, t):
        """
        EDM-style denoiser returning an estimate of x_0 given x_t and sigma=t.

        Args:
            xt: (B, N, D) positions at noise level t
            t:  (B,) positive noise levels (sigma)
        Returns:
            x0_hat: (B, N, D) estimate of x_0 with zero center-of-mass
        """
        assert_mean_zero(xt)

        n_batch = xt.shape[0]

        edges = cast_edges2batch(self._edges_dict, self.edges, n_batch, self._n_particles)
        edges = [edges[0].to(xt.device), edges[1].to(xt.device)]
        # edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles)
        # edges = [edges[0].to(xt.device), edges[1].to(xt.device)]

        c_in = 1 / (t ** 2 + self.sigma_data ** 2) ** 0.5
        x = xt * c_in[:, None, None]
        x = x.reshape(n_batch * self._n_particles, self._n_dimension)

        time_emb = self.time_embedding(torch.log(t) / 4)  # shape (bs, time_embedding_dim)
        time_emb_flatten = time_emb.unsqueeze(1).repeat(1, self._n_particles, 1)
        time_emb_flatten = time_emb_flatten.view(n_batch * self._n_particles, -1)

        if self.net_name == 'egnn':
            if self.model_config['atom_type_embedding_dim'] > 0:
                atom_type_emb = self.atom_type_embedding_layer(self.atom_type_labels)
                atom_type_emb = atom_type_emb.unsqueeze(0).repeat(n_batch, 1, 1)
                atom_type_emb = atom_type_emb.view(n_batch * self._n_particles, -1)

                h = torch.cat([time_emb_flatten, atom_type_emb], dim=1)
            else:
                h = time_emb_flatten

            if self.dataset == 'aldp':
                node_indices0 = edges[0] % self._n_particles
                node_indices1 = edges[1] % self._n_particles
                bond_mask = self.adj[node_indices0, node_indices1].unsqueeze(1)
                distance_sq = torch.sum((x[edges[0]] - x[edges[1]]) ** 2, dim=1, keepdim=True)
                edge_attr = torch.cat([distance_sq, bond_mask], dim=1)

            else:
                edge_attr = None

            _, x_pred = self.model(h, x, edges, edge_attr=edge_attr)

        else:
            raise ValueError('Model not supported')

        x_pred = x_pred.view(n_batch, self._n_particles, self._n_dimension)
        x_pred = remove_mean(x_pred)

        c_skip = self.sigma_data ** 2 / (t ** 2 + self.sigma_data ** 2)
        c_out = t * self.sigma_data / (self.sigma_data ** 2 + t ** 2) ** 0.5

        out = xt * c_skip[:, None, None] + x_pred * c_out[:, None, None]

        return out

    def compute_loss(self, x0):
        """
        Loss implemented according to EDM paper.
        """
        num_samples = x0.shape[0]
        x0 = remove_mean(x0)

        x0 = x0 * self.normalizing_constant

        random_t = torch.exp(torch.randn(num_samples,) * 1.2 - 1.2).to(self.device)
        noise = sample_center_gravity_zero_gaussian(x0.shape, self.device)  # zero gravity

        xt = x0 + random_t[:, None, None] * noise

        x_pred = self.forward(xt, random_t)

        loss_weight = (self.sigma_data ** 2 + random_t ** 2) / (random_t * self.sigma_data) ** 2
        loss = torch.mean(torch.sum((x_pred - x0) ** 2, dim=(1, 2)) * loss_weight)

        return loss

    def ddpm_sampler(
        self,
        num_steps: int,
        true_target,                         # distribution with .log_prob on x0-space
        num_samples: int = 2000,
        init_x: Optional[torch.Tensor] = None,
        time_steps: Optional[torch.Tensor] = None,
        proposal: Literal["scalar","fullcov"]="scalar",
        cov_mats: Optional[List[torch.Tensor]] = None,  # required if proposal="fullcov"
        nus: Optional[List[float]] = None,              # optional per-step scalar multipliers
        tune_time_steps: bool = False,
        progress_bar: bool = False,
    ):
        """
        Importance-sampling DDPM reverse chain with scalar or full-cov proposals.
        Returns (x_samples, normalized_weights, ess, log_weights).
        """
        eps, T = self.eps, self.T
        if time_steps is None:
            time_steps = self._time_steps(num_steps, schedule="geom", eps=eps, T=T)

        if init_x is None:
            x = sample_center_gravity_zero_gaussian((num_samples, self._n_particles, self._n_dimension),
                                                    device=self.device) * T
        else:
            x = init_x.detach().clone()

        y = x.detach().clone()

        if tune_time_steps:
            cov0 = (torch.tensor(T, device=self.device)**2) * self._identity_subspace()
            log_w = -self._log_cg_gauss(torch.zeros_like(x), cov0, y, full=True)
        else:
            log_w = -self._log_cg_gauss(torch.zeros_like(x), torch.tensor(T, device=self.device)**2, y, full=False)

        rng = tqdm(range(len(time_steps)-1, 0, -1), desc='DDPM', leave=False) if progress_bar \
            else range(len(time_steps)-1, 0, -1)

        for n in rng:
            tn  = time_steps[n]
            tn1 = time_steps[n-1]
            t_batch = tn * torch.ones(x.shape[0], device=self.device)

            x0_hat = self.forward(x, t_batch)
            mu = self._reverse_mu(x, x0_hat, tn, tn1)

            # proposal
            if proposal == "scalar":
                sigma2 = self._proposal_sigma2_scalar(tn, tn1)
                if nus is not None:
                    sigma2 = sigma2 * nus[n-1]
                if tune_time_steps:
                    Sigma = sigma2 * self._identity_subspace()
                    x = self._sample_cg_gauss(mu, Sigma, full=True)
                    log_w -= self._log_cg_gauss(mu, Sigma, x, full=True)
                else:
                    x = self._sample_cg_gauss(mu, sigma2, full=False)
                    log_w -= self._log_cg_gauss(mu, sigma2, x, full=False)
            elif proposal == "fullcov":
                assert cov_mats is not None, "cov_mats must be provided for proposal='fullcov'."
                Sigma = cov_mats[n-1]
                x = self._sample_cg_gauss(mu, Sigma, full=True)
                log_w -= self._log_cg_gauss(mu, Sigma, x, full=True)
            else:
                raise ValueError("proposal must be 'scalar' or 'fullcov'")

            # target bridge term
            if tune_time_steps:
                cov_tar = (tn**2 - tn1**2) * self._identity_subspace()
                log_w += self._log_cg_gauss(x, cov_tar, y, full=True)
            else:
                sigma2_tar = tn**2 - tn1**2
                log_w += self._log_cg_gauss(x, sigma2_tar, y, full=False)

            y = x.detach().clone()

        log_w += true_target.log_prob(y / self.normalizing_constant)

        with torch.no_grad():
            w = torch.exp(log_w - log_w.max())
            w = w / w.sum()
            ess = 1.0 / (w**2).sum()

        return x / self.normalizing_constant, w, ess, log_w


    def ddim_sampler(
        self,
        num_steps,
        true_target,
        num_samples=2000,
        time_steps=None,
        progress_bar=False,
        *,
        divergence_mode: str = "exact",      # "exact" | "hutchinson_vjp" | "hutchinson_jvp"
        num_trace_probes: int = 1,
        probe_kind: str = "rademacher",      # "rademacher" | "gaussian"
        reuse_probes_across_time: bool = True,
    ):
        """
        Reverse-time deterministic (DDIM-style) sampler with log-density correction
        via ∫ div f dt, supporting exact trace, Hutchinson-VJP, or Hutchinson-JVP.

        Returns:
            x_eps:   (M, P, D) samples in data scale (divided by normalizing_constant)
            w:       (M,) normalized importance weights
            ess:     scalar effective sample size
            log_w:   (M,) unnormalized log-weights
        """
        # --- validation ---
        if divergence_mode not in {"exact", "hutchinson_vjp", "hutchinson_jvp"}:
            raise ValueError(f"Unknown divergence_mode: {divergence_mode}")
        if divergence_mode != "exact" and num_trace_probes < 1:
            raise ValueError("num_trace_probes must be >= 1 for Hutchinson modes.")
        if probe_kind.lower() not in {"rademacher", "rad", "gaussian", "normal", "gauss"}:
            raise ValueError(f"Unknown probe kind: {probe_kind}")

        eps, T = self.eps, self.T
        if time_steps is None:
            rho = 1/7
            time_steps = (eps**rho +
                        torch.linspace(0, 1, num_steps, device=self.device) * (T**rho - eps**rho)
                        ) ** (1/rho)

        # 1) sample and get log p_T
        x = sample_center_gravity_zero_gaussian(
                (num_samples, self._n_particles, self._n_dimension), device=self.device
            ) * T
        log_prior = center_gravity_zero_gaussian_log_likelihood(
            torch.zeros_like(x), torch.tensor(T, device=self.device, dtype=x.dtype)**2, x
        )  # (M,)

        log_det = torch.zeros(num_samples, device=self.device, dtype=x.dtype)
        M, P, Dp = num_samples, self._n_particles, self._n_dimension
        Dtot = P * Dp

        # Pre-sample probe vectors (optional reuse across time)
        probe_bank = None
        if divergence_mode != "exact" and reuse_probes_across_time:
            probe_bank = [
                _rand_probe((M, Dtot), probe_kind, device=self.device, dtype=x.dtype)
                for _ in range(num_trace_probes)
            ]

        it = (tqdm(range(len(time_steps)-1, 0, -1), desc='DDIM')
            if progress_bar else range(len(time_steps)-1, 0, -1))

        for n in it:
            t_n  = time_steps[n]      # larger
            t_nm = time_steps[n-1]    # smaller
            dt   = (t_nm - t_n)       # NOTE: negative

            # re-attach for autograd
            x = x.detach().requires_grad_(True)      # shape (M, P, Dp)

            # 1) compute the drift f(x, t_n)
            t_vec  = torch.full((M,), fill_value=float(t_n), device=x.device, dtype=x.dtype)
            x0_hat = self.forward(x, t_vec)
            f      = -(x0_hat - x) / t_n             # EDM parametrization

            # 2) divergence estimate (note: "exact" recomputes forward internally)
            if divergence_mode == "exact":
                div_f = _estimate_divergence_exact(self, x, t_n)                # (M,)
            elif divergence_mode == "hutchinson_vjp":
                div_f = _estimate_divergence_hutch_vjp(
                            self, x, t_n, f=f,
                            num_probes=num_trace_probes,
                            probe_kind=probe_kind,
                            reuse_vecs=probe_bank
                        )
            else:  # "hutchinson_jvp"
                div_f = _estimate_divergence_hutch_jvp(
                            self, x, t_n,
                            num_probes=num_trace_probes,
                            probe_kind=probe_kind,
                            reuse_vecs=probe_bank
                        )

            # 3) accumulate integral of divergence
            with torch.no_grad():
                log_det = (log_det + div_f * dt).detach()  # dt<0 → minus sign overall

            # 4) Euler step
            x = (x + dt * f).detach()

        # 5) final importance weights
        x_eps      = x / self.normalizing_constant
        log_target = true_target.log_prob(x_eps)
        log_prop   = log_prior - log_det            # == log p_T + ∫_0^T div f dt
        log_w      = log_target - log_prop

        with torch.no_grad():
            w   = torch.exp(log_w - torch.max(log_w))
            w  /= w.sum()
            ess = 1.0 / (w**2).sum()

        return x_eps, w, ess, log_w


    def forward_ess_ddim(
        self,
        x0,
        log_prob_x0,
        num_steps,
        time_steps=None,
        progress_bar=True,
        *,
        divergence_mode: str = "exact",      # "exact" | "hutchinson_vjp" | "hutchinson_jvp"
        num_trace_probes: int = 1,
        probe_kind: str = "rademacher",
        reuse_probes_across_time: bool = True,
    ):
        """
        Forward-time deterministic (DDIM-style) flow from x0 to xT, estimating
        log p0(x0) ≈ log pT(xT) + ∫_0^T div f dt, and then computing importance
        weights w ∝ p_data(x0) / p_flow(x0).

        Returns:
            ess:   scalar ESS for the forward weights
            log_w: (M,) unnormalized log-weights
        """
        # --- validation ---
        if divergence_mode not in {"exact", "hutchinson_vjp", "hutchinson_jvp"}:
            raise ValueError(f"Unknown divergence_mode: {divergence_mode}")
        if divergence_mode != "exact" and num_trace_probes < 1:
            raise ValueError("num_trace_probes must be >= 1 for Hutchinson modes.")
        if probe_kind.lower() not in {"rademacher", "rad", "gaussian", "normal", "gauss"}:
            raise ValueError(f"Unknown probe kind: {probe_kind}")

        eps, T = self.eps, self.T
        if time_steps is None:
            rho = 1/7
            time_steps = (eps**rho +
                        torch.linspace(0, 1, num_steps, device=self.device) * (T**rho - eps**rho)
                        ) ** (1/rho)

        x = (x0 * self.normalizing_constant).detach()
        x = x.requires_grad_(True)

        log_det = torch.zeros(x0.shape[0], device=self.device, dtype=x.dtype)
        M, P, Dp = x0.shape[0], self._n_particles, self._n_dimension
        Dtot = P * Dp

        # Correct iterator
        it = (tqdm(range(len(time_steps)-1), desc='Forward DDIM')
            if progress_bar else range(len(time_steps)-1))

        # Pre-sample probes (optional reuse along time)
        probe_bank = None
        if divergence_mode != "exact" and reuse_probes_across_time:
            probe_bank = [
                _rand_probe((M, Dtot), probe_kind, device=self.device, dtype=x.dtype)
                for _ in range(num_trace_probes)
            ]

        for n in it:
            t_n  = time_steps[n]      # smaller
            t_np = time_steps[n+1]    # larger
            dt   = (t_np - t_n)       # NOTE: positive

            # Rebuild graph each step
            x = x.detach().requires_grad_(True)

            t_vec  = torch.full((M,), fill_value=float(t_n), device=x.device, dtype=x.dtype)
            x0_hat = self.forward(x, t_vec)
            f      = -(x0_hat - x) / t_n

            if divergence_mode == "exact":
                div_f = _estimate_divergence_exact(self, x, t_n)
            elif divergence_mode == "hutchinson_vjp":
                div_f = _estimate_divergence_hutch_vjp(
                            self, x, t_n, f=f,
                            num_probes=num_trace_probes,
                            probe_kind=probe_kind,
                            reuse_vecs=probe_bank
                        )
            else:  # "hutchinson_jvp"
                div_f = _estimate_divergence_hutch_jvp(
                            self, x, t_n,
                            num_probes=num_trace_probes,
                            probe_kind=probe_kind,
                            reuse_vecs=probe_bank
                        )

            with torch.no_grad():
                log_det = (log_det + div_f * dt).detach()  # dt>0

            x = (x + dt * f).detach()

        log_prior = center_gravity_zero_gaussian_log_likelihood(
            torch.zeros_like(x), torch.tensor(T, device=x.device, dtype=x.dtype)**2, x
        )
        log_prop = log_prior + log_det      # ≈ log p0(x0) induced by the flow
        log_tar  = log_prob_x0

        log_w = log_tar - log_prop
        with torch.no_grad():
            ess = compute_forward_ess(log_w)

        return ess, log_w


    def estimate_forward_ess(
        self,
        x0: torch.Tensor,
        log_prob_x0: torch.Tensor,
        num_steps: int,
        time_steps: Optional[torch.Tensor] = None,
        proposal: Literal["scalar","fullcov"]="scalar",
        cov_mats: Optional[List[torch.Tensor]] = None,   # req if proposal="fullcov"
        nus: Optional[List[float]] = None,               # optional per-step scalar multipliers
        tune_time_steps: bool = False,
        alpha: float = 2.0,
        progress_bar: bool = False,
    ):
        """
        Forward (data->noise) ESS estimator via importance weights.

        log_w = log p(x0) + sum_t log p(x_{t+1}|x_t) - sum_t log q(x_t|x_{t+1}) - log p_T(x_T)
        """
        eps, T = self.eps, self.T
        if time_steps is None:
            time_steps = self._time_steps(num_steps, schedule="geom", eps=eps, T=T)

        x = x0 * self.normalizing_constant
        y = x.clone()
        log_w = log_prob_x0.clone()
        M = x0.shape[0]

        rng = tqdm(range(len(time_steps)-1), desc='Forward ESS', leave=False) if progress_bar \
            else range(len(time_steps)-1)

        for n in rng:
            tn  = time_steps[n]
            tn1 = time_steps[n+1]

            # target forward diffusion step: x ~ N(y, (tn1^2 - tn^2) I_subspace)
            if tune_time_steps:
                cov_tar = (tn1**2 - tn**2) * self._identity_subspace()
                x = self._sample_cg_gauss(y, cov_tar, full=True)
                log_w += self._log_cg_gauss(y, cov_tar, x, full=True)
            else:
                sigma2_tar = tn1**2 - tn**2
                x = y + sample_center_gravity_zero_gaussian(y.shape, device=self.device) * (sigma2_tar**0.5)
                log_w += self._log_cg_gauss(y, sigma2_tar, x, full=False)

            # reverse proposal q(y | x)
            t_batch = tn1 * torch.ones(M, device=self.device)
            x0_hat = self.forward(x, t_batch)
            mu = self._reverse_mu(x, x0_hat, tn1, tn)  # mean that maps from x_{n+1} to x_n

            if proposal == "scalar":
                sigma2 = self._proposal_sigma2_scalar(tn1, tn)
                if nus is not None:
                    sigma2 = sigma2 * nus[n]
                if tune_time_steps:
                    Sigma = sigma2 * self._identity_subspace()
                    log_w -= self._log_cg_gauss(mu, Sigma, y, full=True)
                else:
                    log_w -= self._log_cg_gauss(mu, sigma2, y, full=False)
            elif proposal == "fullcov":
                assert cov_mats is not None, "cov_mats must be provided for proposal='fullcov'."
                Sigma = cov_mats[n]
                log_w -= self._log_cg_gauss(mu, Sigma, y, full=True)
            else:
                raise ValueError("proposal must be 'scalar' or 'fullcov'")

            y = x.clone()

        # subtract terminal prior log-density
        if tune_time_steps:
            cov_final = (time_steps[-1]**2) * self._identity_subspace()
            log_w -= self._log_cg_gauss(torch.zeros_like(x), cov_final, x, full=True)
        else:
            log_w -= self._log_cg_gauss(torch.zeros_like(x), time_steps[-1]**2, x, full=False)

        with torch.no_grad():
            forward_ess = compute_forward_ess(log_w)

        # optional alpha divergence (same definition as your code)
        if alpha == 1:
            alpha_div = torch.mean(log_w)
        else:
            alpha_div = torch.logsumexp(log_w * (alpha - 1), dim=0)

        return alpha_div, forward_ess, log_w


    # ---------------------------
    # small utilities / helpers
    # ---------------------------
    def _time_steps(self, num_steps: int, schedule: Literal["geom", "karras"]="geom",
                    eps: Optional[float]=None, T: Optional[float]=None, rho: float=1/7) -> torch.Tensor:
        """Generate monotonically increasing time steps in [eps, T]."""
        eps = torch.tensor(self.eps if eps is None else eps, device=self.device, dtype=torch.float32)
        T   = torch.tensor(self.T   if T   is None else T,   device=self.device, dtype=torch.float32)
        if schedule == "geom":
            ts = torch.tensor(np.geomspace(float(eps), float(T), num_steps), device=self.device, dtype=torch.float32)
        elif schedule == "karras":
            ts = (eps**rho + torch.linspace(0, 1, num_steps, device=self.device)*(T**rho - eps**rho))**(1/rho)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        return ts

    def _log_cg_gauss(self, mu: torch.Tensor, cov, x: torch.Tensor, full: bool) -> torch.Tensor:
        """Log-likelihood under center-of-gravity Gaussians."""
        if full:
            return center_of_gravity_gaussian_log_likelihood_full_cov(mu, cov, x, dataset=self.dataset)
        else:
            # cov is scalar sigma^2 in the zero-gravity utility
            return center_gravity_zero_gaussian_log_likelihood(mu, cov, x)

    def _sample_cg_gauss(self, mu: torch.Tensor, cov, full: bool) -> torch.Tensor:
        """Sampling from center-of-gravity Gaussians."""
        if full:
            return sample_center_of_gravity_gaussian_full_cov(mu, cov, dataset=self.dataset)
        else:
            z = sample_center_gravity_zero_gaussian(mu.shape, device=self.device)
            return mu + torch.sqrt(cov) * z

    def _proposal_sigma2_scalar(self, tn: torch.Tensor, tn1: torch.Tensor) -> torch.Tensor:
        """Scalar variance used in EDM DDPM reverse kernel."""
        return (tn1**2 * (tn**2 - tn1**2)) / (tn**2)

    def _reverse_mu(self, x: torch.Tensor, x0_hat: torch.Tensor, tn: torch.Tensor, tn1: torch.Tensor) -> torch.Tensor:
        """Reverse DDPM mean."""
        return (tn1/tn)**2 * x + (1 - (tn1/tn)**2) * x0_hat

    def _drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """DDIM drift field f(x,t) for your parametrization."""
        x0_hat = self.forward(x, t)
        return -(x0_hat - x) / t

    @torch.no_grad()
    def _identity_subspace(self) -> torch.Tensor:
        return torch.eye(self.subspace_dim, device=self.device)
    

# Helpers for divergence estimation
def _rand_probe(shape, kind: str, device, dtype):
    k = kind.lower()
    if k in ("rademacher", "rad"):
        # ±1 with prob 0.5
        return (torch.randint(0, 2, shape, device=device, dtype=torch.int64) * 2 - 1).to(dtype)
    if k in ("gaussian", "normal", "gauss"):
        return torch.randn(shape, device=device, dtype=dtype)
    raise ValueError(f"Unknown probe kind: {kind}")

def _estimate_divergence_exact(model, x, t_scalar):
    """
    Exact divergence via full Jacobian trace:
      div f(x) = trace(∂f/∂x)
    NOTE: O(M * D^2) memory — use only for small D.
    """
    M, P, D = x.shape
    Dtot = P * D
    x_flat = x.view(M, Dtot)

    def f_flat_single(xf):
        x_unf = xf.view(P, D)
        t1 = torch.as_tensor(t_scalar, device=xf.device, dtype=xf.dtype).expand(1)
        s  = model.forward(x_unf.unsqueeze(0), t1).squeeze(0)
        f_unf = -(s - x_unf) / t_scalar
        return f_unf.reshape(-1)

    J = vmap(jacrev(f_flat_single))(x_flat)      # (M, Dtot, Dtot)
    return torch.diagonal(J, dim1=1, dim2=2).sum(dim=1)  # (M,)

def _estimate_divergence_hutch_vjp(model, x, t_scalar, f=None, *, num_probes=1, probe_kind="rademacher", reuse_vecs=None):
    """
    Hutchinson via VJP: div ≈ E_v [ v^T (J v) ] by computing J^T v with a single autograd.grad.
    - If f is provided, reuses it to avoid an extra forward pass.
    - reuse_vecs: list of pre-sampled probes (each (M, Dtot)) or None.
    """
    M, P, D = x.shape
    Dtot = P * D

    if f is None:
        t_vec = torch.as_tensor(t_scalar, device=x.device, dtype=x.dtype).expand(M)
        x0_hat = model.forward(x, t_vec)
        f = -(x0_hat - x) / t_scalar
    f_flat = f.view(M, Dtot)

    div_est = torch.zeros(M, device=x.device, dtype=x.dtype)

    for k in range(num_probes):
        v = reuse_vecs[k] if reuse_vecs is not None else _rand_probe((M, Dtot), probe_kind, device=x.device, dtype=x.dtype)
        dot = (f_flat * v).sum()   # sums over batch+dim; autograd stays per-sample
        (JTv,) = torch.autograd.grad(dot, x, retain_graph=(k < num_probes - 1), create_graph=False)
        div_est = div_est + (JTv.view(M, Dtot) * v).sum(dim=1)

    return div_est / num_probes  # (M,)

def _estimate_divergence_hutch_jvp(model, x, t_scalar, *, num_probes=1, probe_kind="rademacher", reuse_vecs=None):
    """
    Hutchinson via JVP: div ≈ E_v [ v^T (J v) ] with jvp on a per-sample function.
    This recomputes forward inside jvp (by design).
    """
    M, P, D = x.shape
    Dtot = P * D
    x_flat = x.view(M, Dtot)

    def f_flat_single(xf):
        x_unf = xf.view(P, D)
        t1 = torch.as_tensor(t_scalar, device=xf.device, dtype=xf.dtype).expand(1)
        s  = model.forward(x_unf.unsqueeze(0), t1).squeeze(0)
        f_unf = -(s - x_unf) / t_scalar
        return f_unf.reshape(-1)

    div_est = torch.zeros(M, device=x.device, dtype=x.dtype)

    for k in range(num_probes):
        v = reuse_vecs[k] if reuse_vecs is not None else _rand_probe((M, Dtot), probe_kind, device=x.device, dtype=x.dtype)
        # Batched JVP: per-sample jvp(f_i, (x_i,), (v_i,))
        Jv = vmap(lambda xi, vi: jvp(f_flat_single, (xi,), (vi,))[1])(x_flat, v)
        div_est = div_est + (Jv * v).sum(dim=1)

    return div_est / num_probes  # (M,)
