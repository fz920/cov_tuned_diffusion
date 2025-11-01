import torch
from torch.autograd.functional import jacobian
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from torchcfm.optimal_transport import OTPlanSampler
from torch.func import vmap, jacrev

from .utils import assert_mean_zero, remove_mean, sample_center_gravity_zero_gaussian, center_gravity_zero_gaussian_log_likelihood, create_edges, cast_edges2batch, TimeEmbedding_Diffusion, center_of_gravity_gaussian_log_likelihood_full_cov, sample_center_of_gravity_gaussian_full_cov, compute_forward_ess
from .egnn import EGNN

from torchdiffeq import odeint

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
        E[x_eps|x_t]
        """
        # assert_mean_zero(xt)

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

    def compute_loss(self, x0, use_ot=False):
        """
        Loss implemented according to EDM paper.
        """
        num_samples = x0.shape[0]
        x0 = remove_mean(x0)

        x0 = x0 * self.normalizing_constant

        random_t = torch.exp(torch.randn(num_samples,) * 1.2 - 1.2).to(self.device)
        noise = sample_center_gravity_zero_gaussian(x0.shape, self.device)  # zero gravity

        if use_ot:
            with torch.no_grad():
                x0_before = x0.view(num_samples, -1)
                noise_before = noise.view(num_samples, -1)

                sampler = OTPlanSampler(method="exact")
                x0_after, noise_after = sampler.sample_plan(x0_before, noise_before, replace=False)

                x0 = x0_after.view(num_samples, self._n_particles, self._n_dimension)
                noise = noise_after.view(num_samples, self._n_particles, self._n_dimension)

        xt = x0 + random_t[:, None, None] * noise

        x_pred = self.forward(xt, random_t)

        loss_weight = (self.sigma_data ** 2 + random_t ** 2) / (random_t * self.sigma_data) ** 2
        loss = torch.mean(torch.sum((x_pred - x0) ** 2, dim=(1, 2)) * loss_weight)

        return loss

    def ddpm_sampler(self, num_steps, true_target, num_samples=2000, init_x=None,
                     nus=None, time_steps=None, progress_bar=False, tune_time_steps=False):

        eps, T = self.eps, self.T
        if time_steps is None:
            time_steps = torch.tensor(np.geomspace(eps, T, num_steps), dtype=torch.float32).to(self.device)

        if init_x is None:
            x = sample_center_gravity_zero_gaussian((num_samples, self._n_particles, self._n_dimension), device=self.device) * T
        else:
            x = init_x.detach().clone()

        y = x.detach().clone()
        if tune_time_steps:
            log_w = -center_of_gravity_gaussian_log_likelihood_full_cov(
                torch.zeros_like(x, device=self.device), torch.tensor(T) ** 2 * torch.eye(self.subspace_dim, device=self.device), y,
                dataset=self.dataset
            )
        else:
            log_w = -center_gravity_zero_gaussian_log_likelihood(torch.zeros_like(x, device=self.device), torch.tensor(T) ** 2, y)

        if progress_bar:
            bar = tqdm(range(len(time_steps)-1, 0, -1), desc='DDPM Progress')
        else:
            bar = range(len(time_steps)-1, 0, -1)
        for n in bar:
            tn = time_steps[n]
            tn1 = time_steps[n-1]

            hat_x0 = self.forward(x, tn * torch.ones(num_samples, device=self.device))

            sigma2 = (tn1 ** 2 * (tn ** 2 - tn1 ** 2)) / tn ** 2
            if nus is not None:
                nu = nus[n-1]
                # sigma2_sde = tn ** 2 - tn1 ** 2
                # sigma2 = (1 - nu) * sigma2_sde + nu * sigma2
                sigma2 = nu * sigma2

            mu =  (tn1 / tn) ** 2 * x + (1 - (tn1 / tn) ** 2) * hat_x0
            if tune_time_steps:
                x = sample_center_of_gravity_gaussian_full_cov(
                    mu, sigma2 * torch.eye(self.subspace_dim, device=self.device), dataset=self.dataset
                )
            else:
                z = sample_center_gravity_zero_gaussian((num_samples, self._n_particles, self._n_dimension), device=self.device)
                x = mu + torch.sqrt(sigma2) * z

            # cov_tar = (tn ** 2 - tn1 ** 2) * torch.eye(self.subspace_dim, device=self.device)
            if tune_time_steps:
                log_w += center_of_gravity_gaussian_log_likelihood_full_cov(
                    x, (tn ** 2 - tn1 ** 2) * torch.eye(self.subspace_dim, device=self.device), y, dataset=self.dataset
                )
                log_w -= center_of_gravity_gaussian_log_likelihood_full_cov(
                    mu, sigma2 * torch.eye(self.subspace_dim, device=self.device), x, dataset=self.dataset
                )
            else:
                log_w += center_gravity_zero_gaussian_log_likelihood(x, tn ** 2 - tn1 ** 2, y)  # target
                log_w -= center_gravity_zero_gaussian_log_likelihood(mu, sigma2, x)  # proposal

            y = x.detach().clone()

        log_w += true_target.log_prob(y / self.normalizing_constant)

        w = torch.exp(log_w - torch.max(log_w))
        w = w / torch.sum(w)

        ess = 1 / torch.sum(w ** 2)
        return x / self.normalizing_constant, w, ess, log_w
    
    # def ddim_sampler(self, num_steps, true_target, num_samples=2000, init_x=None,
    #                  nus=None, time_steps=None, progress_bar=False, tune_time_steps=False):

    #     x = sample_center_gravity_zero_gaussian((num_samples, self._n_particles, self._n_dimension), device=self.device) * T


    #     w    = torch.exp(logw - logw.max())
    #     w    = w / w.sum()

    #     ess  = 1. / (w**2).sum()

    #     return x_eps, w, ess

    def ddim_sampler(self, num_steps, true_target, num_samples=2000,
                    time_steps=None, progress_bar=False):
        eps, T = self.eps, self.T
        if time_steps is None:
            rho = 1/7
            time_steps = (eps**rho +
                        torch.linspace(0, 1, num_steps, device=self.device)*(T**rho - eps**rho)
                        ) ** (1/rho)

        # 1) sample and get log p_T
        x = sample_center_gravity_zero_gaussian((num_samples,
                                                self._n_particles,
                                                self._n_dimension),
                                                device=self.device) * T
        log_prior = center_gravity_zero_gaussian_log_likelihood(
                        torch.zeros_like(x), torch.tensor(T)**2, x
                    )  # shape: (M,)

        log_det = torch.zeros(num_samples, device=self.device)
        M, D = num_samples, self._n_particles*self._n_dimension

        it = (tqdm(range(len(time_steps)-1, 0, -1), desc='DDIM')
            if progress_bar else
            range(len(time_steps)-1, 0, -1))

        for n in it:
            t_n  = time_steps[n]
            t_nm = time_steps[n-1]
            dt   = (t_nm - t_n)

            # re-attach for autograd
            x = x.detach().requires_grad_(True)      # shape (M, P, D)

            # 1) compute the drift f(x, t_n)
            x0_hat = self.forward(x, t_n*torch.ones(M, device=x.device))
            f     = -(x0_hat - x)/t_n                  # your parametrization

            # 2) flatten for jacobian‐of‐f
            M, P, D = x.shape
            Dtot    = P*D
            x_flat  = x.view(M, Dtot)

            def f_flat(xf):
                x_unf = xf.view(P, D)
                s = self.forward(
                    x_unf.unsqueeze(0),
                    torch.tensor([t_n], device=xf.device),
                ).squeeze(0)
                f_unf = -(s - x_unf)/t_n
                return f_unf.view(-1)

            # 3) build the full Jacobian J[i] = ∂f_flat/∂x_flat at each sample i
            #    J has shape (M, Dtot, Dtot)
            J = vmap(jacrev(f_flat))(x_flat)

            with torch.no_grad():
                # 4) trace out the divergence and accumulate
                #    div[f] = sum_k ∂f_k / ∂x_k  = trace(J)
                div_f = torch.diagonal(J, dim1=1, dim2=2).sum(dim=1)  # shape (M,)
                log_det = (log_det + div_f * dt).detach()

            #     # 5) do your actual Euler step *without* grad
            x = (x + dt * f).detach()

        # 3) final weights
        x_eps     = x / self.normalizing_constant
        # log_scale = - D * torch.log(torch.tensor(self.normalizing_constant,
        #                                         device=x.device))
        log_target= true_target.log_prob(x_eps)
        log_prop  = log_prior - log_det

        log_w = log_target - log_prop
        with torch.no_grad():
            w     = torch.exp(log_w - torch.max(log_w))
            w    /= w.sum()
            ess   = 1.0 / (w**2).sum()

        return x_eps, w, ess, log_w

    def forward_ess_ddim(self, x0, log_prob_x0, num_steps, time_steps=None,
                         progress_bar=True):
        eps, T = self.eps, self.T
        if time_steps is None:
            rho = 1/7
            time_steps = (eps**rho +
                        torch.linspace(0, 1, num_steps, device=self.device)*(T**rho - eps**rho)
                        ) ** (1/rho)

        x = x0 * self.normalizing_constant

        log_det = torch.zeros(x0.shape[0], device=self.device)
        M, D = x0.shape[0], self._n_particles*self._n_dimension

        it = (tqdm(range(len(time_steps)-1), desc='Forward DDIM')
            if progress_bar else
            range(range(len(time_steps)-1)))

        for n in it:
            t_n  = time_steps[n]
            t_nm = time_steps[n+1]
            dt   = (t_nm - t_n)

            # re-attach for autograd
            x = x.detach().requires_grad_(True)      # shape (M, P, D)

            # 1) compute the drift f(x, t_n)
            x0_hat = self.forward(x, t_n*torch.ones(M, device=x.device))
            f     = -(x0_hat - x)/t_n                  # your parametrization

            # 2) flatten for jacobian‐of‐f
            M, P, D = x.shape
            Dtot    = P*D
            x_flat  = x.view(M, Dtot)

            def f_flat(xf):
                x_unf = xf.view(P, D)
                s = self.forward(
                    x_unf.unsqueeze(0),
                    torch.tensor([t_n], device=xf.device),
                ).squeeze(0)
                f_unf = -(s - x_unf)/t_n
                return f_unf.view(-1)

            # 3) build the full Jacobian J[i] = ∂f_flat/∂x_flat at each sample i
            #    J has shape (M, Dtot, Dtot)
            J = vmap(jacrev(f_flat))(x_flat)

            with torch.no_grad():
                # 4) trace out the divergence and accumulate
                #    div[f] = sum_k ∂f_k / ∂x_k  = trace(J)
                div_f = torch.diagonal(J, dim1=1, dim2=2).sum(dim=1)  # shape (M,)
                log_det = (log_det + div_f * dt).detach()

            #     # 5) do your actual Euler step *without* grad
            x = (x + dt * f).detach()

        log_prior = center_gravity_zero_gaussian_log_likelihood(
                        torch.zeros_like(x), torch.tensor(T)**2, x
                    )
        log_prop = log_prior + log_det
        log_tar = log_prob_x0

        log_w = log_tar - log_prop
        with torch.no_grad():
            ess = compute_forward_ess(log_w)

        return ess, log_w


    def ddpm_sampler_model(self, num_steps, true_target, num_samples=2000,
                            init_x=None, cov_model=None, progress_bar=False,
                            time_steps=None, lams=None, output_scale=None, diag=False):

        eps, T = self.eps, self.T
        if time_steps is None:
            time_steps = torch.tensor(np.geomspace(eps, T, num_steps), dtype=torch.float32).to(self.device)

        if init_x is None:
            x = sample_center_gravity_zero_gaussian((num_samples, self._n_particles, self._n_dimension), device=self.device) * T
        else:
            x = init_x.detach().clone()

        y = x.detach().clone()
        cov_prop = torch.tensor(T) ** 2 * torch.eye(self.subspace_dim, device=self.device)

        # log_w = -mvn_log_density(torch.zeros_like(x, device=self.device), cov_prop, y)
        log_w = -center_of_gravity_gaussian_log_likelihood_full_cov(
            torch.zeros_like(x, device=self.device), cov_prop, x, dataset=self.dataset
        )

        if progress_bar:
            bar = tqdm(range(len(time_steps)-1, 0, -1), desc='DDPM Low Rank Progress')
        else:
            bar = range(len(time_steps)-1, 0, -1)
        for n in bar:
            tn = time_steps[n]
            tn1 = time_steps[n-1]

            hat_x0 = self.forward(x, tn * torch.ones(num_samples, device=self.device))


            sigma2_ddpm = tn1 ** 2 * (tn ** 2 - tn1 ** 2) / tn ** 2
            sigma2_full, sigma2_sub = cov_model.get_cov_mat(x, tn1*torch.ones(num_samples, device=self.device), sigma2_ddpm=sigma2_ddpm, lam=lams[n-1], auto_diff=False,
                                                            output_scale=output_scale, diag=diag)
            
            mu = (tn1 / tn) ** 2 * x + (1 - (tn1 / tn) ** 2) * hat_x0

            x = sample_center_of_gravity_gaussian_full_cov(mu, sigma2_sub, dataset=self.dataset)
            log_w -= center_of_gravity_gaussian_log_likelihood_full_cov(mu, sigma2_sub, x, dataset=self.dataset)

            cov_tar = (tn ** 2 - tn1 ** 2) * torch.eye(self.subspace_dim, device=self.device)
            log_w += center_of_gravity_gaussian_log_likelihood_full_cov(x, cov_tar, y, dataset=self.dataset)

            y = x.detach().clone()

        log_w += true_target.log_prob(y / self.normalizing_constant)

        w = torch.exp(log_w - torch.max(log_w))
        w = w / torch.sum(w)

        ess = 1 / torch.sum(w ** 2)

        return x / self.normalizing_constant, w, ess, log_w
    
    def ddpm_sampler_low_rank(self, num_steps, true_target, num_samples=2000,
                                init_x=None, cov_mat_all=None, time_steps=None,
                                progress_bar=False):
 
        eps, T = self.eps, self.T
        if time_steps is None:
            time_steps = torch.tensor(np.geomspace(eps, T, num_steps), dtype=torch.float32).to(self.device)

        if init_x is None:
            x = sample_center_gravity_zero_gaussian((num_samples, self._n_particles, self._n_dimension), device=self.device) * T
        else:
            x = init_x.detach().clone()

        y = x.detach().clone()
        cov_prop = torch.tensor(T) ** 2 * torch.eye(self.subspace_dim, device=self.device)

        # log_w = -mvn_log_density(torch.zeros_like(x, device=self.device), cov_prop, y)
        log_w = -center_of_gravity_gaussian_log_likelihood_full_cov(
            torch.zeros_like(x, device=self.device), cov_prop, x, dataset=self.dataset
        )

        if progress_bar:
            bar = tqdm(range(len(time_steps)-1, 0, -1), desc='DDPM Low Rank Progress')
        else:
            bar = range(len(time_steps)-1, 0, -1)
        for n in bar:
            tn = time_steps[n]
            tn1 = time_steps[n-1]

            hat_x0 = self.forward(x, tn * torch.ones(num_samples, device=self.device))

            sigma2_prop = cov_mat_all[n-1]
            mu = (tn1 / tn) ** 2 * x + (1 - (tn1 / tn) ** 2) * hat_x0

            x = sample_center_of_gravity_gaussian_full_cov(mu, sigma2_prop, dataset=self.dataset)

            log_w -= center_of_gravity_gaussian_log_likelihood_full_cov(mu, sigma2_prop, x, dataset=self.dataset)

            cov_tar = (tn ** 2 - tn1 ** 2) * torch.eye(self.subspace_dim, device=self.device)
            log_w += center_of_gravity_gaussian_log_likelihood_full_cov(x, cov_tar, y, dataset=self.dataset)

            y = x.detach().clone()

        log_w += true_target.log_prob(y / self.normalizing_constant)

        w = torch.exp(log_w - torch.max(log_w))
        w = w / torch.sum(w)

        ess = 1 / torch.sum(w ** 2)

        return x / self.normalizing_constant, w, ess, log_w

    def est_forward_ess(self, x0, log_prob_x0, num_steps, time_steps=None,
                        nus=None, progress_bar=True, alpha=2, tune_time_steps=False):
        eps, T = self.eps, self.T

        # geometric spacing
        if time_steps is None:
            time_steps = torch.tensor(np.geomspace(eps, T, num_steps), dtype=torch.float32).to(self.device)

        x = x0.clone() * self.normalizing_constant
        y = x.clone()
        log_w = log_prob_x0.clone()

        num_samples = x0.shape[0]

        if progress_bar:
            bar = tqdm(range(len(time_steps)-1), desc='Forward ESS Progress')
        else:
            bar = range(len(time_steps)-1)
        for n in bar:
            tn = time_steps[n]
            tn1 = time_steps[n+1]

            sigma2_tar = tn1 ** 2 - tn ** 2

            if tune_time_steps:
                sigma2_tar = sigma2_tar * torch.eye(self.subspace_dim, device=self.device)
                x = sample_center_of_gravity_gaussian_full_cov(y, sigma2_tar, dataset=self.dataset)
                log_w += center_of_gravity_gaussian_log_likelihood_full_cov(y, sigma2_tar, x)
            else:
                x = x + sample_center_gravity_zero_gaussian(x.shape, device=self.device) * sigma2_tar ** 0.5
                log_w += center_gravity_zero_gaussian_log_likelihood(y, sigma2_tar, x)

            hat_x0 = self.forward(x, tn1 * torch.ones(num_samples, device=self.device))

            sigma2 = (tn ** 2 * (tn1 ** 2 - tn ** 2)) / tn1 ** 2
            if nus is not None:
                nu = nus[n]
                # sigma2 = (1 - nu) * sigma2_tar + nu * sigma2
                sigma2 = nu * sigma2

            mu = (tn / tn1) ** 2 * x + (1 - (tn / tn1) ** 2) * hat_x0

            if tune_time_steps:
                sigma2 = sigma2 * torch.eye(self.subspace_dim, device=self.device)
                log_w -= center_of_gravity_gaussian_log_likelihood_full_cov(mu, sigma2, y)
            else:
                log_w -= center_gravity_zero_gaussian_log_likelihood(mu, sigma2, y)
            y = x.clone()

        if tune_time_steps:
            log_w -= center_of_gravity_gaussian_log_likelihood_full_cov(torch.zeros_like(x, device=self.device),
                                                                time_steps[-1] ** 2 * torch.eye(self.subspace_dim, device=self.device), x)
        else:
            log_w -= center_gravity_zero_gaussian_log_likelihood(torch.zeros_like(x, device=self.device),
                                                                    time_steps[-1] ** 2, x)            

        # alpha divergence
        if alpha == 1:
            alpha_div = torch.mean(log_w)
        else:
            alpha_div = torch.logsumexp(log_w*(alpha-1), dim=0)

        with torch.no_grad():
            forward_ess = compute_forward_ess(log_w)

        return alpha_div, forward_ess, log_w

    def forward_ess_model(self, x0, log_prob_x0, num_steps, eps=0.002, T=80.0,
                         alpha=2, time_steps=None, cov_model=None, progress_bar=False,
                         lams=None, output_scale=None, diag=False):
        if time_steps is None:
            eps, T = self.eps, self.T
            time_steps = torch.tensor(np.geomspace(eps, T, num_steps), dtype=torch.float32).to(self.device)

        x = x0.clone() * self.normalizing_constant
        y = x.clone()
        log_w = log_prob_x0.clone()

        num_samples = x0.shape[0]

        if progress_bar:
            bar = tqdm(range(len(time_steps)-1), desc='Forward ESS Progress (Low Rank)')
        else:
            bar = range(len(time_steps)-1)
        for n in bar:
            tn = time_steps[n]
            tn1 = time_steps[n+1]

            cov_tar = tn1 ** 2 - tn ** 2
            x = y + sample_center_gravity_zero_gaussian(x.shape, device=self.device) * cov_tar ** 0.5

            log_w += center_of_gravity_gaussian_log_likelihood_full_cov(
                y, cov_tar * torch.eye(self.subspace_dim, device=self.device),
                x, dataset=self.dataset
                )

            hat_x0 = self.forward(x, tn1 * torch.ones(num_samples, device=self.device))

            sigma2_ddpm = (tn ** 2 * (tn1 ** 2 - tn ** 2)) / tn1 ** 2
            sigma2_full, sigma2_sub = cov_model.get_cov_mat(x, tn1*torch.ones(num_samples, device=self.device), sigma2_ddpm=sigma2_ddpm, lam=lams[n], auto_diff=False,
                                                            output_scale=output_scale, diag=diag)

            mu = (tn / tn1) ** 2 * x + (1 - (tn / tn1) ** 2) * hat_x0
            log_w -= center_of_gravity_gaussian_log_likelihood_full_cov(
                mu, sigma2_sub, y, dataset=self.dataset
                )

            y = x.clone()

        cov_final = torch.tensor(T) ** 2 * torch.eye(self.subspace_dim, device=self.device)
        log_w -= center_of_gravity_gaussian_log_likelihood_full_cov(
            torch.zeros_like(x, device=self.device), cov_final, x, dataset=self.dataset
            )

        # Z_inv = torch.mean(torch.exp(-log_w))
        # forward_ess = num_samples ** 2 / (torch.sum(torch.exp(log_w)) * Z_inv)
        with torch.no_grad():
            forward_ess = compute_forward_ess(log_w)

        if alpha == 1:
            alpha_div = torch.mean(log_w)
        else:
            alpha_div = torch.logsumexp(log_w * (alpha - 1), dim=0)

        return alpha_div, forward_ess, log_w

    def forward_ess_low_rank(self, x0, log_prob_x0, num_steps, eps=0.002, T=80.0,
                              alpha=2, time_steps=None, cov_mat_all=None, progress_bar=False):
         if time_steps is None:
             time_steps = torch.tensor(np.geomspace(eps, T, num_steps), dtype=torch.float32).to(self.device)
 
         x = x0.clone() * self.normalizing_constant
         y = x.clone()
         log_w = log_prob_x0.clone()
 
         num_samples = x0.shape[0]
 
         if progress_bar:
             bar = tqdm(range(len(time_steps)-1), desc='Forward ESS Progress (Low Rank)')
         else:
             bar = range(len(time_steps)-1)
         for n in bar:
             tn = time_steps[n]
             tn1 = time_steps[n+1]
 
             cov_tar = tn1 ** 2 - tn ** 2
             x = y + sample_center_gravity_zero_gaussian(x.shape, device=self.device) * cov_tar ** 0.5
 
             log_w += center_of_gravity_gaussian_log_likelihood_full_cov(
                 y, cov_tar * torch.eye(self.subspace_dim, device=self.device),
                 x, dataset=self.dataset
                 )
 
             hat_x0 = self.forward(x, tn1 * torch.ones(num_samples, device=self.device))
 
             sigma2_mat = cov_mat_all[n]
 
             mu = (tn / tn1) ** 2 * x + (1 - (tn / tn1) ** 2) * hat_x0
             log_w -= center_of_gravity_gaussian_log_likelihood_full_cov(
                 mu, sigma2_mat, y, dataset=self.dataset
                 )
 
             y = x.clone()
 
         cov_final = torch.tensor(T) ** 2 * torch.eye(self.subspace_dim, device=self.device)
         log_w -= center_of_gravity_gaussian_log_likelihood_full_cov(
             torch.zeros_like(x, device=self.device), cov_final, x, dataset=self.dataset
             )
 
         # Z_inv = torch.mean(torch.exp(-log_w))
         # forward_ess = num_samples ** 2 / (torch.sum(torch.exp(log_w)) * Z_inv)
         with torch.no_grad():
             forward_ess = compute_forward_ess(log_w)
 
         if alpha == 1:
             alpha_div = torch.mean(log_w)
         else:
             alpha_div = torch.logsumexp(log_w * (alpha - 1), dim=0)
 
         return alpha_div, forward_ess, log_w
