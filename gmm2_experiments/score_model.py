import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from utils import MyMLP
from cov_model import CovModel

class ScoreNet(nn.Module):
    def __init__(self, input_dim, hidden_size=128, n_layers=3, emb_size=128,
                 time_emb="sinusoidal", device='cuda', cov_form='isotropic'):
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

    def ddpm_sampler_low_rank_model(self, num_steps, num_samples=2000, eps=0.002, T=80.0,
                     true_gmm=None, init_x=None, cov_pred_model=None, sigma=None):

        time_steps = torch.tensor(np.geomspace(eps, T, num_steps), dtype=torch.float32).to(self.device)
        if init_x is None:
            x = torch.randn(num_samples, self.input_dim, device=self.device) * T
        else:
            x = init_x.detach().clone()

        y = x.detach().clone()
        cov_prop = torch.tensor(T) ** 2 * torch.eye(self.input_dim, device=self.device)
        log_w = -mvn_log_density(torch.zeros_like(x, device=self.device), cov_prop, y)

        for n in tqdm(range(len(time_steps)-1, 0, -1), desc='DDPM Progress'):
            tn = time_steps[n]
            tn1 = time_steps[n-1]

            hat_x0 = self.forward(x, tn * torch.ones(num_samples, device=self.device))

            A_pred = cov_pred_model(tn * torch.ones(num_samples, 1, device=self.device), x)
            sigma2 = torch.bmm(A_pred, A_pred.transpose(1, 2)) + sigma[n-1] * torch.eye(self.input_dim, device=self.device).unsqueeze(0).repeat(num_samples, 1, 1)

            sigma2_ddpm = (tn1 ** 2 * (tn ** 2 - tn1 ** 2)) / tn ** 2
            sigma2 = sigma2 * sigma2_ddpm

            # assert sigma2.shape == (self.input_dim, self.input_dim), "Invalid sigma2 shape {}".format(sigma2.shape)

            mu = (tn1 / tn) ** 2 * x + (1 - (tn1 / tn) ** 2) * hat_x0

            z1 = torch.randn(num_samples, A_pred.shape[2], device=self.device)
            z2 = torch.randn_like(x, device=self.device)
            # x = mu + torch.sqrt(sigma2) * z
            # sigma2_L = torch.linalg.cholesky(sigma2)
            # x = mu + z @ sigma2_L.T

            # efficient sampling
            # A_pred has shape (num_samples, input_dim, k)
            # z1 has shape (num_samples, k)
            x = mu + torch.einsum('nk,nik->ni', z1, sigma2_ddpm ** 0.5 * A_pred) + (sigma[n-1] * sigma2_ddpm) ** 0.5 * z2

            log_w -= mvn_log_density(mu, sigma2, x)

            cov_tar = (tn ** 2 - tn1 ** 2) * torch.ones(num_samples, self.input_dim, device=self.device)
            cov_tar = torch.diag_embed(cov_tar)
            log_w += mvn_log_density(x, cov_tar, y)

            y = x.detach().clone()

        log_w += true_gmm.log_prob(x)

        w = torch.exp(log_w - torch.max(log_w))
        w = w / torch.sum(w)

        ess = 1 / torch.sum(w ** 2)

        return x, w, ess

    def ddpm_sampler(self, num_steps, num_samples=2000, eps=0.002, T=80.0,
                     true_gmm=None, init_x=None, cov_params=None, time_steps=None,
                     cov_form=None, progress_bar=False):
        if cov_form is None:
            cov_form = self.cov_form

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

    def forward_ess_low_rank_model(self, x0, log_prob_x0, num_steps, eps=0.002, T=80.0,
                             cov_pred_model=None, sigma=None, alpha=2):
        time_steps = torch.tensor(np.geomspace(eps, T, num_steps), dtype=torch.float32).to(self.device)
        x = x0.clone()
        y = x.clone()
        log_w = log_prob_x0.clone()

        num_samples = x0.shape[0]

        for n in range(len(time_steps)-1):
            tn = time_steps[n]
            tn1 = time_steps[n+1]

            x = x + (tn1 ** 2 - tn ** 2) ** 0.5 * torch.randn_like(x, device=self.device)
            cov_tar = (tn1 ** 2 - tn ** 2) * torch.eye(self.input_dim, device=self.device)

            log_w += mvn_log_density(y, cov_tar, x)

            hat_x0 = self.forward(x, tn1 * torch.ones(num_samples, device=self.device))

            A_pred = cov_pred_model(tn1 * torch.ones(num_samples, 1, device=self.device), x)
            sigma2 = torch.bmm(A_pred, A_pred.transpose(1, 2)) + sigma[n] * torch.eye(self.input_dim, device=self.device).unsqueeze(0).repeat(num_samples, 1, 1)

            sigma2_ddpm = (tn ** 2 * (tn1 ** 2 - tn ** 2)) / tn1 ** 2
            sigma2 = sigma2 * sigma2_ddpm

            mu = (tn / tn1) ** 2 * x + (1 - (tn / tn1) ** 2) * hat_x0

            log_w -= mvn_log_density(mu, sigma2, y)

            y = x.clone()

        cov_final = torch.tensor(T) ** 2 * torch.eye(self.input_dim, device=self.device)

        log_w -= mvn_log_density(torch.zeros_like(x, device=self.device), cov_final, x)

        Z_inv = torch.mean(torch.exp(-log_w))
        forward_ess = num_samples ** 2 / (torch.sum(torch.exp(log_w)) * Z_inv)

        if alpha == 1:
            alpha_div = torch.mean(log_w)
        else:
            alpha_div = torch.logsumexp(log_w * (alpha - 1), dim=0)

        return alpha_div, forward_ess


class CovariancePredictor(nn.Module):
    def __init__(self, input_dim, hidden_size=64, n_layers=1, condition_on_x=False,
                 k=None):
        """
        Args:
            input_dim (int): Dimensionality of the covariance matrix (i.e. the number of rows/columns).
            hidden_size (int): Hidden layer dimension.
            condition_on_x (bool): If True, the model expects a concatenated [t, x_t] input.
        """
        super(CovariancePredictor, self).__init__()
        # If conditioning on x, the input dimension increases (e.g., 1 for t + input_dim for x_t)
        in_features = 1 + input_dim if condition_on_x else 1
        
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = []
        for _ in range(n_layers - 1):
            self.fc2.append(nn.Linear(hidden_size, hidden_size))
        self.fc2 = nn.ModuleList(self.fc2)
        
        if k is None or k == 0:
            self.k = self.input_dim
        else:
            self.k = k

        self.out_dim = input_dim * self.k
        self.fc3 = nn.Linear(hidden_size, self.out_dim)

        # initialize the output layer so that cov prediction is close to 0
        torch.nn.init.uniform_(self.fc3.weight, -1e-3, 1e-3)
        torch.nn.init.constant_(self.fc3.bias, 0.0)

        self.input_dim = input_dim
        self.condition_on_x = condition_on_x

    def forward(self, t, x_t=None):
        """
        Args:
            t (Tensor): A tensor of shape (batch_size, 1) representing time.
            x_t (Tensor): (Optional) A tensor of shape (batch_size, input_dim) representing the state.
        Returns:
            L (Tensor): A lower-triangular matrix of shape (batch_size, input_dim, input_dim)
                        such that covariance = L @ L^T.
        """
        # If conditioning on x, concatenate it with t.
        if self.condition_on_x:
            if x_t is None:
                raise ValueError("x_t must be provided when condition_on_x=True.")

            inp = torch.cat([t, x_t], dim=-1)
        else:
            inp = t  # Only using time

        h = F.relu(self.fc1(inp))
        for layer in self.fc2:
            h = F.relu(layer(h))
        # h = F.relu(self.fc2(h))
        params = self.fc3(h)

        cov_pred = params.view(-1, self.input_dim, self.k)

        return cov_pred
    

class MeanPredictor(nn.Module):
    def __init__(self, input_dim, hidden_size=64, n_layers=1, condition_on_x=False):
        """
        Args:
            input_dim (int): Dimensionality of the covariance matrix (i.e. the number of rows/columns).
            hidden_size (int): Hidden layer dimension.
            condition_on_x (bool): If True, the model expects a concatenated [t, x_t] input.
        """
        super(CovariancePredictor, self).__init__()
        # If conditioning on x, the input dimension increases (e.g., 1 for t + input_dim for x_t)
        in_features = 1 + input_dim if condition_on_x else 1

        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = []
        for _ in range(n_layers - 1):
            self.fc2.append(nn.Linear(hidden_size, hidden_size))
        self.fc2 = nn.ModuleList(self.fc2)

        self.out_dim = input_dim
        self.fc3 = nn.Linear(hidden_size, self.out_dim)

        # initialize the output layer so that cov prediction is close to 0
        torch.nn.init.uniform_(self.fc3.weight, -1e-3, 1e-3)
        torch.nn.init.constant_(self.fc3.bias, 0.0)

        self.input_dim = input_dim
        self.condition_on_x = condition_on_x

    def forward(self, t, x_t=None):
        """
        Args:
            t (Tensor): A tensor of shape (batch_size, 1) representing time.
            x_t (Tensor): (Optional) A tensor of shape (batch_size, input_dim) representing the state.
        Returns:
            L (Tensor): A lower-triangular matrix of shape (batch_size, input_dim, input_dim)
                        such that covariance = L @ L^T.
        """
        # If conditioning on x, concatenate it with t.
        if self.condition_on_x:
            if x_t is None:
                raise ValueError("x_t must be provided when condition_on_x=True.")

            inp = torch.cat([t, x_t], dim=-1)
        else:
            inp = t  # Only using time

        h = F.relu(self.fc1(inp))
        for layer in self.fc2:
            h = F.relu(layer(h))
        # h = F.relu(self.fc2(h))
        params = self.fc3(h)

        return params


def mvn_log_density(mean, cov, samples):
    mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
    log_density = mvn.log_prob(samples)
    return log_density


def project_to_pd(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Projects a square matrix M onto the set of positive-definite matrices.
    
    Args:
        M (torch.Tensor): a 2D square tensor (N x N).
        eps (float): the minimum eigenvalue threshold.
        
    Returns:
        M_pd (torch.Tensor): the nearest positive-definite matrix in terms of 
                             Frobenius norm (via eigenvalue clamping).
    """
    # Ensure M is symmetric
    Msym = 0.5 * (M + M.transpose(-1, -2))
    
    # Eigen-decomposition of the symmetric matrix
    # (torch.linalg.eigh / torch.symeig both work for symmetric/Hermitian matrices)
    w, V = torch.linalg.eigh(Msym)
    
    # Clamp eigenvalues to be at least eps
    w_clamped = torch.clamp(w, min=eps)
    
    # Reconstruct the matrix: M_pd = V * w_clamped * V^T
    # We multiply each eigenvector by its (clamped) eigenvalue, then re-form
    M_pd = (V * w_clamped) @ V.transpose(-1, -2)
    
    return M_pd

def multivariate_normal_log_pdf_single(mean, covariance, x):
    """
    Compute the log-pdf of a multivariate normal distribution under the simplification that covariance is a scalar.
    """
    # print(x.shape)
    num_samples, input_dim = x.shape
    diff = x - mean
    norm_squared = torch.sum(diff ** 2, dim=1)
    first_term = -0.5 * norm_squared / covariance
    second_term = -0.5 * input_dim * torch.log(2 * torch.pi * covariance)
    return first_term + second_term

# # def multivariate_normal_log_pdf(means, covariances, x):
# #     """
# #     Compute the log-pdf of a multivariate normal distribution where both the mean and covariance
# #     vary for each data point.

# #     Parameters:
# #     - means: Tensor of shape (num_samples, input_dim), the means for each sample.
# #     - covariances: Tensor of shape (num_samples, input_dim, input_dim), the covariance matrices for each sample.
# #     - x: Tensor of shape (num_samples, input_dim), the data points for which to compute the log-pdf.

# #     Returns:
# #     - log_pdf: Tensor of shape (num_samples,), the log-pdf for each sample.
# #     """
# #     num_samples, input_dim = x.shape
    
# #     # Compute the difference for each sample
# #     diff = x - means  # Shape: (num_samples, input_dim)
    
# #     # Precompute the determinants and inverses for the covariance matrices
# #     cov_dets = torch.linalg.det(covariances)  # Shape: (num_samples,)
# #     cov_invs = torch.linalg.inv(covariances)  # Shape: (num_samples, input_dim, input_dim)
    
# #     # Compute the Mahalanobis distance for each sample
# #     mahalanobis_terms = torch.einsum('bi,bij,bj->b', diff, cov_invs, diff)  # Shape: (num_samples,)
    
# #     # Compute the normalization constant for each sample
# #     normalization_consts = -0.5 * input_dim * torch.log(2 * torch.pi) - 0.5 * torch.log(cov_dets)  # Shape: (num_samples,)
    
# #     # Compute the log-pdf for each sample
# #     log_pdf = -0.5 * mahalanobis_terms + normalization_consts  # Shape: (num_samples,)
# #     return log_pdf

# # covariance does not depend on data
# def multivariate_normal_log_pdf(means, covariance, x):
#     """
#     Compute the log-pdf of a multivariate normal distribution where the covariance is shared across all samples.

#     Parameters:
#     - means: Tensor of shape (num_samples, input_dim), the means for each sample.
#     - covariance: Tensor of shape (input_dim, input_dim), the shared covariance matrix for all samples.
#     - x: Tensor of shape (num_samples, input_dim), the data points for which to compute the log-pdf.

#     Returns:
#     - log_pdf: Tensor of shape (num_samples,), the log-pdf for each sample.
#     """
#     num_samples, input_dim = x.shape

#     # Compute the difference for each sample
#     diff = x - means  # Shape: (num_samples, input_dim)

#     # Compute the inverse and determinant of the shared covariance matrix
#     # print(covariance)
#     cov_inv = torch.linalg.inv(covariance)  # Shape: (input_dim, input_dim)
#     cov_det = torch.linalg.det(covariance)  # Scalar

#     # Compute the Mahalanobis distance for each sample
#     mahalanobis_terms = torch.einsum('bi,ij,bj->b', diff, cov_inv, diff)  # Shape: (num_samples,)

#     # Compute the normalization constant (same for all samples)
#     normalization_const = -0.5 * input_dim * np.log(2 * np.pi) - 0.5 * torch.log(cov_det)  # Scalar

#     # Compute the log-pdf for each sample
#     log_pdf = -0.5 * mahalanobis_terms + normalization_const  # Shape: (num_samples,)
#     return log_pdf


def test_equivalence():
    """
    Test whether the two PDF functions agree in the special
    case where covariance is scalar * I and the mean is identical
    for all samples.
    """
    torch.manual_seed(100)            # For reproducibility

    num_samples = 100
    input_dim = 100

    # Create a single mean (1D vector)
    mean = torch.randn(num_samples, input_dim)  # shape: (input_dim,)

    # Create a scalar covariance > 0
    covariance_scalar = torch.tensor(0.7)  # for example

    # Build the corresponding covariance matrix: sigma^2 * I
    covariance_matrix = covariance_scalar * torch.ones(num_samples, input_dim)
    covariance_matrix = torch.diag_embed(covariance_matrix)  # shape: (num_samples, input_dim, input_dim)

    # Generate random data
    x = torch.randn(num_samples, input_dim)

    # For the first function, we need a mean per sample, but all identical:
    # means = mean.unsqueeze(0).expand(num_samples, -1)  # shape: (num_samples, input_dim)

    # Compute log-pdfs
    log_pdf_multivariate = mvn_log_density(mean, covariance_matrix, x)
    log_pdf_single = multivariate_normal_log_pdf_single(mean, covariance_scalar, x)

    # Print and compare
    print("Log PDF (multivariate):", log_pdf_multivariate)
    print("Log PDF (single):      ", log_pdf_single)
    print("Max absolute difference:", torch.max(torch.abs(log_pdf_multivariate - log_pdf_single)).item())

if __name__ == "__main__":
    test_equivalence()
