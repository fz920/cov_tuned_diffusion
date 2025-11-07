"""Dataset and target-distribution loading helpers."""

from __future__ import annotations

import numpy as np
import torch

from .target_distributions import (
    load_aldp,
    AldpEnergy,
    LennardJonesEnergy,
    MultiDoubleWellEnergy,
)
from ..models.utils import remove_mean
from ..utils.path_config import get_dataset_path


def load_dataset(dataset: str = "aldp", device: str | torch.device = "cuda", partition: str = "train") -> torch.Tensor:
    """Load a dataset split and cast it to a centered torch tensor."""
    dataset = dataset.lower()
    data_path = get_dataset_path(dataset, partition)

    if dataset == "dw4":
        training_data = np.load(data_path)
        training_data = np.reshape(training_data, (training_data.shape[0], 4, 2))
    elif dataset == "lj13":
        training_data = np.load(data_path)
        training_data = np.reshape(training_data, (training_data.shape[0], 13, 3))
    elif dataset == "lj55":
        training_data = np.load(data_path)
        training_data = np.reshape(training_data, (training_data.shape[0], 55, 3))
    elif dataset == "aldp":
        training_dataset = load_aldp(train_path=data_path, train_n_points=int(1e6))[0]
        training_data = training_dataset["positions"]
        training_data = np.reshape(training_data, (training_data.shape[0], 22, 3))
    else:
        raise ValueError(f"Unknown dataset '{dataset}'.")

    if not isinstance(training_data, torch.Tensor):
        training_data = torch.tensor(training_data, device=device).float()

    training_data = training_data.to(device)
    training_data = remove_mean(training_data)
    return training_data


def load_target_dist(dataset: str):
    """Return the energy function associated with a dataset identifier."""
    dataset = dataset.lower()
    if dataset == "dw4":
        return MultiDoubleWellEnergy(dimensionality=8, n_particles=4)
    if dataset == "lj13":
        return LennardJonesEnergy(dimensionality=39, n_particles=13)
    if dataset == "lj55":
        return LennardJonesEnergy(dimensionality=165, n_particles=55)
    if dataset == "aldp":
        return AldpEnergy(temperature=300.0)
    raise ValueError(f"Unknown dataset '{dataset}'.")


__all__ = ["load_dataset", "load_target_dist"]
