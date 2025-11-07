import torch
import torch.distributions as D

class GMM:
    def __init__(self, means, covariances, weights, device):
        self.means = means
        self.covariances = covariances
        self.weights = weights
        self.device = device
        self.input_dim = means[0].shape[0]

        self.gmm = self.create_gmm(self.weights, self.means, self.covariances)

    def create_gmm(self, weights=None, means=None, covariances=None):
        mixture_distribution = D.Categorical(weights)
        component_distributions = D.Independent(D.Normal(
            torch.stack(means),
            torch.stack([torch.sqrt(torch.diag(cov)) for cov in covariances])
        ), 1)

        return D.MixtureSameFamily(mixture_distribution, component_distributions)

    def log_prob(self, x):
        return self.gmm.log_prob(x)

    def sample(self, num_samples):
        return self.gmm.sample((num_samples,))

    def score(self, x, t):
        z = x.detach().clone()
        z.requires_grad = True

        new_covariances = []
        for covariance in self.covariances:
            new_covariances.append(covariance.to(self.device) + t ** 2 * torch.eye(self.input_dim, device=self.device))

        new_gmm = self.create_gmm(self.weights, self.means, new_covariances)
        log_densities = new_gmm.log_prob(z)
        # print(z.requires_grad)
        # print(log_densities.requires_grad)

        gradients = torch.autograd.grad(log_densities.sum(), z)[0]
        return gradients

    def hess_batch(self, x, t):
        # x is of shape (batch_size, input_dim)
        hessians = []
        for i in range(x.shape[0]):
            # For each sample, ensure it requires grad.
            x_single = x[i].detach().clone().requires_grad_(True)
            hess_single = torch.autograd.functional.hessian(
                lambda z: self.create_gmm(
                    self.weights, 
                    self.means, 
                    [cov.to(self.device) + t**2 * torch.eye(self.input_dim, device=self.device)
                    for cov in self.covariances]
                ).log_prob(z),
                x_single
            )
            hessians.append(hess_single)
        return torch.stack(hessians, dim=0)

def create_gmm(input_dim, device):
    means = [torch.ones(input_dim, device=device) * 1, torch.ones(input_dim, device=device) * -2]
    covariances = [torch.eye(input_dim, device=device) * 0.15 for _ in range(2)]
    weights = torch.tensor([2/3, 1/3], device=device)

    gmm = GMM(means, covariances, weights, device=device)

    return gmm
