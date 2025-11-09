import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utils import MyMLP

class ScoreNet(nn.Module):
    def __init__(self, input_dim, hidden_size=128, n_layers=3, emb_size=128,
                 time_emb="sinusoidal", device='cuda', cov_form=None):
        super().__init__()
        self.mlp = MyMLP(
            hidden_size=hidden_size,
            hidden_layers=n_layers,
            emb_size=emb_size,
            out_dim=input_dim,
            time_emb=time_emb,
            input_emb=time_emb,
            add_t_emb=False,
            concat_t_emb=True,
            input_dim=input_dim
        )

        self.sigma_data = 1.45
        self.device = device
        self.input_dim = input_dim

        self.eps = 0.002
        self.T = 80.0

        self.cov_form = cov_form

    def forward(self, x, t):
        c_in = 1 / (self.sigma_data ** 2 + t ** 2) ** 0.5

        h = self.mlp(x * c_in[:, None], torch.log(t) / 4)

        c_skip = self.sigma_data ** 2 / (t ** 2 + self.sigma_data ** 2)
        c_out = t * self.sigma_data / (self.sigma_data ** 2 + t ** 2) ** 0.5

        return x * c_skip[:, None] + h * c_out[:, None]

    def compute_loss(self, x0):
        num_samples = x0.shape[0]
        random_t = torch.exp(torch.randn(num_samples,) * 1.2 - 1.2).to(self.device)
        xt = x0 + random_t[:, None] * torch.randn_like(x0, device=self.device)
        x_pred = self.forward(xt, random_t)
        loss_weight = (self.sigma_data ** 2 + random_t ** 2) / (random_t * self.sigma_data) ** 2
        loss = torch.mean(torch.sum((x_pred - x0) ** 2, dim=-1) * loss_weight)
        return loss

    def ddpm_sampler(self, num_steps, num_samples=2000, eps=0.002, T=80.0,
                     true_gmm=None, init_x=None, cov_params=None, time_steps=None,
                     cov_form=None, progress_bar=False):
        if time_steps is None:
            time_steps = torch.tensor(np.geomspace(eps, T, num_steps), dtype=torch.float32).to(self.device)

        if init_x is None:
            x = torch.randn(num_samples, self.input_dim, device=self.device) * T
        else:
            x = init_x.detach().clone()

        y = x.detach().clone()
        cov_prop = torch.tensor(T) ** 2 * torch.eye(self.input_dim, device=self.device)
        log_w = -mvn_log_density(torch.zeros_like(x, device=self.device), cov_prop, y)

        bar = tqdm(range(len(time_steps)-1, 0, -1), desc='DDPM Progress') if progress_bar else range(len(time_steps)-1, 0, -1)
        for n in bar:
            tn = time_steps[n]
            tn1 = time_steps[n-1]

            hat_x0 = self.forward(x, tn * torch.ones(num_samples, device=self.device))

            sigma2_ddpm = (tn1 ** 2 * (tn ** 2 - tn1 ** 2)) / tn ** 2
            if cov_form == 'isotropic':
                sigma2 = cov_params[n-1] * sigma2_ddpm * torch.eye(self.input_dim, device=self.device)
            elif cov_form == 'diagonal':
                sigma2 = torch.diag_embed(cov_params[n-1] * sigma2_ddpm)
            elif cov_form == 'full':
                A = cov_params['A'][n-1]
                lam = cov_params['lam'][n-1]
                sigma2 = sigma2_ddpm * (A @ A.T + lam ** 2 * torch.eye(self.input_dim, device=self.device))
            else:
                sigma2 = sigma2_ddpm * torch.eye(self.input_dim, device=self.device)

            mu = (tn1 / tn) ** 2 * x + (1 - (tn1 / tn) ** 2) * hat_x0

            z = torch.randn_like(x, device=self.device)

            sigma2_L = torch.linalg.cholesky(sigma2)
            x = mu + z @ sigma2_L.T

            log_w -= mvn_log_density(mu, sigma2, x)

            cov_tar = (tn ** 2 - tn1 ** 2) * torch.eye(self.input_dim, device=self.device)
            log_w += mvn_log_density(x, cov_tar, y)

            y = x.detach().clone()

        log_w += true_gmm.log_prob(x)

        w = torch.exp(log_w - torch.max(log_w))
        w = w / torch.sum(w)

        ess = 1 / torch.sum(w ** 2)

        return x, w, ess

    def est_forward_ess(self, x0, log_prob_x0, num_steps, eps=0.002, T=80.0,
                        cov_params=None, time_steps=None, alpha=2.0, cov_form=None,
                        progress_bar=False):
        if cov_form is None:
            cov_form = self.cov_form

        if time_steps is None:
            time_steps = torch.tensor(np.geomspace(eps, T, num_steps), dtype=torch.float32).to(self.device)

        x = x0.clone()
        y = x.clone()
        log_w = log_prob_x0.clone()

        num_samples = x0.shape[0]

        bar = tqdm(range(len(time_steps)-1), desc='Forward ESS Progress') if progress_bar else range(len(time_steps)-1)
        for n in bar:
            tn = time_steps[n]
            tn1 = time_steps[n+1]

            x = x + torch.randn_like(x, device=self.device) * (tn1 ** 2 - tn ** 2) ** 0.5
            cov_tar = (tn1 ** 2 - tn ** 2) * torch.eye(self.input_dim, device=self.device)
            log_w += mvn_log_density(y, cov_tar, x)

            hat_x0 = self.forward(x, tn1 * torch.ones(num_samples, device=self.device))

            sigma2_ddpm = (tn ** 2 * (tn1 ** 2 - tn ** 2)) / tn1 ** 2
            if cov_form == 'isotropic':
                sigma2 = cov_params[n] * sigma2_ddpm * torch.eye(self.input_dim, device=self.device)
            elif cov_form == 'diagonal':
                sigma2 = torch.diag_embed(cov_params[n] * sigma2_ddpm)
            elif cov_form == 'full':
                # Cov = sigma2_ddpm * (A @ A.T + lambda2 * I)
                A = cov_params['A'][n]
                lam = cov_params['lam'][n]
                sigma2 = sigma2_ddpm * (A @ A.T + lam ** 2 * torch.eye(self.input_dim, device=self.device))
            else:
                sigma2 = sigma2_ddpm * torch.eye(self.input_dim, device=self.device)
    
            mu = (tn / tn1) ** 2 * x + (1 - (tn / tn1) ** 2) * hat_x0

            log_w -= mvn_log_density(mu, sigma2, y)

            y = x.clone()

        final_cov = torch.tensor(T) ** 2 * torch.eye(self.input_dim, device=self.device)
        log_w -= mvn_log_density(torch.zeros_like(x, device=self.device), final_cov, x)


        alpha_div = torch.logsumexp(log_w * (alpha - 1), dim=0)

        Z_inv = torch.mean(torch.exp(-log_w))
        forward_ess = num_samples ** 2 / (torch.sum(torch.exp(log_w)) * Z_inv)

        return alpha_div, forward_ess, log_w

def mvn_log_density(mean, cov, samples):
    mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
    log_density = mvn.log_prob(samples)
    return log_density

