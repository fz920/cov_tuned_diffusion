import torch
from bgflow import MultiDoubleWellPotential
# from hydra.utils import get_original_cwd

# from dem.energies.base_energy_function import BaseEnergyFunction


class MultiDoubleWellEnergy:
    def __init__(
        self,
        dimensionality,
        n_particles,
        device="cuda",
    ):
        self.n_particles = n_particles
        self.n_spatial_dim = dimensionality // n_particles

        self.device = device

        self.multi_double_well = MultiDoubleWellPotential(
            dim=dimensionality,
            n_particles=n_particles,
            a=0.9,
            b=-4,
            c=0,
            offset=4,
            two_event_dims=False,
        )

        # self.multi_double_well = MultiDoubleWellPotential(
        #     dim=dimensionality,
        #     n_particles=n_particles,
        #     a=0,
        #     b=-4,
        #     c=0.9,
        #     offset=4,
        #     two_event_dims=False,
        # )

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of samples under the energy function
        """
        if len(samples.shape) == 3:
            samples = samples.view(samples.shape[0], -1)
        return -self.multi_double_well.energy(samples).squeeze(-1)

    def log_prob(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of samples under the energy function
        """
        if len(samples.shape) == 3:
            samples = samples.view(samples.shape[0], -1)
        return -self.multi_double_well.energy(samples).squeeze(-1)

    def energy(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Compute energy of samples under the energy function
        """
        if len(samples.shape) == 3:
            samples = samples.view(samples.shape[0], -1)
        return self.multi_double_well.energy(samples).squeeze(-1)

    def interatomic_dist(self, x):
        batch_shape = x.shape[: -len(self.multi_double_well.event_shape)]
        x = x.view(*batch_shape, self.n_particles, self.n_spatial_dim)

        # Compute the pairwise interatomic distances
        # removes duplicates and diagonal
        distances = x[:, None, :, :] - x[:, :, None, :]
        distances = distances[
            :,
            torch.triu(torch.ones((self.n_particles, self.n_particles)), diagonal=1) == 1,
        ]
        dist = torch.linalg.norm(distances, dim=-1)
        return dist

