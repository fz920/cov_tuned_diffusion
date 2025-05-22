from target_dist import load_aldp, AldpEnergy, LennardJonesEnergy, MultiDoubleWellEnergy
import numpy as np
import torch
from model import remove_mean, sample_center_gravity_zero_gaussian, construct_R
from utils.path_config import get_dataset_path


def load_dataset(dataset='aldp', device='cuda', partition='train'):
    if dataset == 'dw4':
        data_path = get_dataset_path(dataset, partition)
        training_data = np.load(data_path)
        training_data = np.reshape(training_data, (training_data.shape[0], 4, 2))

    elif dataset == 'lj13':
        data_path = get_dataset_path(dataset, partition)
        training_data = np.load(data_path)
        training_data = np.reshape(training_data, (training_data.shape[0], 13, 3))

    elif dataset == 'lj55':
        data_path = get_dataset_path(dataset, partition)
        training_data = np.load(data_path)
        training_data = np.reshape(training_data, (training_data.shape[0], 55, 3))
    
    elif dataset == 'aldp':
        data_path = get_dataset_path(dataset, partition)
        training_dataset = load_aldp(train_path=data_path, train_n_points=int(1e6))[0]
        training_data = training_dataset['positions']
        training_data = np.reshape(training_data, (training_data.shape[0], 22, 3))

    else:
        raise ValueError('Dataset not recognized.')

    if not isinstance(training_data, torch.Tensor):
        training_data = torch.tensor(training_data, device=device).float()

    training_data = training_data.to(device)
    training_data = remove_mean(training_data)

    return training_data

def load_target_dist(dataset):
    if dataset == 'dw4':
        target_dist = MultiDoubleWellEnergy(dimensionality=8, n_particles=4)
    elif dataset == 'lj13':
        target_dist = LennardJonesEnergy(dimensionality=39, n_particles=13)
    elif dataset == 'lj55':
        target_dist = LennardJonesEnergy(dimensionality=165, n_particles=55)
    elif dataset == 'aldp':
        target_dist = AldpEnergy(temperature=300.)
    else:
        raise ValueError('Dataset not recognized.')
    
    return target_dist

class Queue():
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def gradient_clipping(flow, gradnorm_queue):
    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    max_grad_norm = 1. * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

    # Clips gradient and returns the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        flow.parameters(), max_norm=max_grad_norm, norm_type=2.0)

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    if float(grad_norm) > max_grad_norm:
        print(f'Clipped gradient with value {grad_norm:.1f} '
              f'while allowed {max_grad_norm:.1f}')
    return grad_norm

def get_constrained_params(mus_unconstrained, etas_unconstrained,
                           gammas_unconstrained, nus_unconstrained):
    mus = torch.sigmoid(mus_unconstrained)
    etas = torch.sigmoid(etas_unconstrained)
    gammas = torch.sigmoid(gammas_unconstrained)
    nus = torch.sigmoid(nus_unconstrained)
    return mus, etas, gammas, nus

def initialize_parameters(num_steps, device='cuda', T=80.0, eps=0.0002):
    etas = torch.logit(torch.ones(num_steps-1) * 0.1).to(device)

    # geometric spacing
    r = (T / eps) ** (1 / num_steps)
    mus = torch.zeros(num_steps-2)
    for n in range(2, num_steps):
        mus[n-2] = (r ** (n - 1) - 1) / (r ** n - 1)
    mus = torch.logit(mus).to(device)

    gammas = torch.logit(torch.ones(num_steps-1) * 0.9).to(device)
    nus = torch.logit(torch.ones(num_steps-1) * 0.9).to(device)

    etas = torch.nn.Parameter(etas)
    mus = torch.nn.Parameter(mus)
    gammas = torch.nn.Parameter(gammas)
    nus = torch.nn.Parameter(nus)
    return etas, mus, gammas, nus

def initialize_parameters_abl(num_steps, device='cuda', T=80.0, eps=0.0002,
                              remove_params='etas'):
    if remove_params == 'etas':
        etas = torch.logit(torch.ones(num_steps-1) * 0.0).to(device)
    else:
        etas = torch.logit(torch.ones(num_steps-1) * 0.1).to(device)

    # geometric spacing
    r = (T / eps) ** (1 / num_steps)
    mus = torch.zeros(num_steps-2)
    for n in range(2, num_steps):
        mus[n-2] = (r ** (n - 1) - 1) / (r ** n - 1)
    mus = torch.logit(mus).to(device)

    if remove_params == 'gammas':
        gammas = torch.logit(torch.ones(num_steps-1) * 1).to(device)
    else:
        gammas = torch.logit(torch.ones(num_steps-1) * 0.9).to(device)

    if remove_params == 'nus':
        nus = torch.logit(torch.ones(num_steps-1) * 1).to(device)
    else:
        nus = torch.logit(torch.ones(num_steps-1) * 0.9).to(device)

    if remove_params == 'etas and gammas':
        etas = torch.logit(torch.ones(num_steps-1) * 0.0).to(device)
        gammas = torch.logit(torch.ones(num_steps-1) * 1).to(device)

    etas = torch.nn.Parameter(etas)
    mus = torch.nn.Parameter(mus)
    gammas = torch.nn.Parameter(gammas)
    nus = torch.nn.Parameter(nus)
    return etas, mus, gammas, nus

def get_time_steps(num_steps, etas, gammas, mus, eps=0.0002, T=80, device='cuda'):
    assert len(etas) == num_steps - 1
    assert len(gammas) == num_steps - 1
    assert len(mus) == num_steps - 2
    time_steps = torch.zeros(num_steps).to(device)
    time_steps[0], time_steps[-1] = eps, T
    for n in range(num_steps-1, 1, -1):
        tn = time_steps[n]
        tn1 = (tn - eps) * mus[n-2] + eps
        time_steps[n-1] = tn1

    target_time_points = torch.zeros(num_steps-1,)
    proposal_time_points = torch.zeros(num_steps-1,)

    for n in range(len(time_steps)-1, 0, -1):
        # DDIM to the end point
        tn = time_steps[n]
        tn1 = time_steps[n-1]

        u1 = (tn - tn1) * etas[n-1] + tn1  # t_tilde'
        target_time_points[n-1] = u1

        u2 = (tn - tn1) * gammas[n-1] + tn1
        proposal_time_points[n-1] = u2

    return time_steps, target_time_points, proposal_time_points

def get_ocm_cov(x0, num_steps, num_samples_est_hess, score_model):
    proj_mat = construct_R(score_model._n_particles, score_model._n_dimension,
                           device=score_model.device)
    eps, T = score_model.eps, score_model.T
    time_steps = torch.tensor(np.geomspace(eps, T, num_steps)).to(score_model.device)

    subspace_dim = score_model.subspace_dim
    ocm_cov = torch.zeros(num_steps-1, subspace_dim, subspace_dim).to(score_model.device)
    for n in range(num_steps-1, 0, -1):
        tn, tn1 = time_steps[n], time_steps[n-1]

        x_tn = x0 + tn * sample_center_gravity_zero_gaussian(x0.shape, device=score_model.device)

        x0_hat = score_model(x_tn, tn * torch.ones(num_samples_est_hess, device=score_model.device))
        score_tn = -(x_tn - x0_hat) / (tn ** 2)

        score_tn = score_tn.view(num_samples_est_hess, -1)
        score_tn_proj = score_tn @ proj_mat

        hess_est = -torch.einsum("bi,bj->bij", score_tn_proj, score_tn_proj).mean(0)

        sigma = (tn ** 2 - tn1 ** 2) ** 0.5

        ocm_cov[n-1] = sigma ** 4 * hess_est + sigma ** 2 * torch.eye(subspace_dim, device=score_model.device)

        sigma2_ddpm = (tn1 ** 2 * (tn ** 2 - tn1 ** 2)) / tn ** 2
        ocm_cov[n-1] = ocm_cov[n-1] / sigma2_ddpm

        try:
            torch.linalg.cholesky(ocm_cov[n-1])
        except:
            print("Cholesky failed at time step:", n)
            breakpoint()

    return ocm_cov
